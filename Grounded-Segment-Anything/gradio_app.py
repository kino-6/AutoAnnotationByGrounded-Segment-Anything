import os
import random
import cv2
from scipy import ndimage
import gradio as gr
import argparse
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionInpaintPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
import yaml

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator

# Global variables
sam_predictor = None
sam_automask_generator = None
groundingdino_model = None
inpaint_pipeline = None

# Paths to model checkpoints
sam_checkpoint = "sam_vit_h_4b8939.pth"
config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
output_dir = "outputs"
device = "cuda"
GRADIO_CONFIG_FILE = "config.yaml"
DEFAULT_COLOR_PALETTE = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [128, 128, 128],
    [255, 165, 0],
    [75, 0, 130],
    [238, 130, 238],
    [0, 128, 128],
    [128, 0, 128],
    [128, 128, 0],
    [192, 192, 192],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [220, 20, 60],
    [255, 105, 180],
    [30, 144, 255],
    [0, 191, 255],
    [50, 205, 50],
    [173, 216, 230],
    [255, 69, 0],
    [240, 230, 140],
    [255, 228, 181],
    [244, 164, 96],
    [199, 21, 133],
    [123, 104, 238],
    [0, 100, 0],
    [0, 250, 154],
    [184, 134, 11],
]


def load_config(config_file_path):
    """
    Load configuration from a YAML file.
    """
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as file:
            return yaml.safe_load(file)
    return {
        # Default configuration
        "task_type": "automatic",
        "text_prompt": "",
        "inpaint_prompt": "",
        "box_threshold": 0.3,
        "text_threshold": 0.25,
        "iou_threshold": 0.5,
        "inpaint_mode": "merge",
        "scribble_mode": "split",
        "openai_api_key": "",
        "start_frame": 0,
        "stop_frame": 1000,
        "skip_frame": 1,
        "color_palette": DEFAULT_COLOR_PALETTE
    }


def save_config(config):
    with open(GRADIO_CONFIG_FILE, "w") as file:
        yaml.safe_dump(config, file)


def load_model(config_file, checkpoint_file, device):
    """
    Load and initialize the GroundingDINO model from configuration and checkpoint files.
    """
    args = SLConfig.fromfile(config_file)
    args.device = device
    model = build_model(args)

    # Load the model checkpoint
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Loaded model state:", load_res)

    model.to(device)
    model.eval()

    return model


def load_classes(classes_file_path):
    """
    Load existing class names from classes.txt and return a dictionary mapping class name to ID.
    """
    class_name_to_id = {}
    if os.path.exists(classes_file_path):
        with open(classes_file_path, "r") as f:
            for idx, line in enumerate(f):
                class_name = line.strip()
                class_name_to_id[class_name] = idx
    return class_name_to_id


def update_classes(new_classes, classes_file_path):
    """
    Update classes.txt with any new class names and return updated mapping of class name to ID.
    """
    class_name_to_id = load_classes(classes_file_path)

    with open(classes_file_path, "a") as f:
        for class_name in new_classes:
            if class_name not in class_name_to_id:
                new_id = len(class_name_to_id)
                class_name_to_id[class_name] = new_id
                f.write(f"{class_name}\n")

    return class_name_to_id


def initialize_models():
    """
    Initialize and return all required models.
    """
    global sam_predictor, sam_automask_generator, groundingdino_model

    if sam_predictor is None:
        assert sam_checkpoint, "sam_checkpoint is not found!"
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
        sam_automask_generator = SamAutomaticMaskGenerator(sam)

    if groundingdino_model is None:
        groundingdino_model = load_model(config_file, ckpt_filenmae, device=device)

    return sam_predictor, sam_automask_generator, groundingdino_model


def transform_image(image_pil):
    """
    Transform the input image for model processing.
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    """
    Get grounding outputs from the model.
    """
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    with torch.no_grad():
        outputs = model(image[None].to(device), captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # Filter outputs
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # Get phrases
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def process_image(
    image_pil,
    scribble,
    task_type,
    sam_predictor,
    text_prompt,
    box_threshold,
    text_threshold,
    iou_threshold,
):
    """
    Process the image and return the masks and other details based on task type.
    """
    image = np.array(image_pil)

    if task_type == "scribble":
        masks = process_scribble_task(image, scribble, sam_predictor)
        return masks, None, None
    elif task_type == "automask":
        masks = sam_automask_generator.generate(image)
        return masks, None, None
    else:
        transformed_image = transform_image(image_pil).to(device)
        masks, boxes_filt, pred_phrases = process_other_tasks(
            transformed_image,
            task_type,
            image_pil,
            sam_predictor,
            text_prompt,
            box_threshold,
            text_threshold,
            iou_threshold,
        )

        # If no masks are found, return None to skip further processing
        if masks is None or len(masks) == 0:
            return None, None, None

    return masks, boxes_filt, pred_phrases


def process_scribble_task(image, scribble, sam_predictor):
    """
    Process scribble task to generate masks.
    """
    sam_predictor.set_image(image)
    scribble = scribble.convert("RGB")
    scribble = np.array(scribble)
    scribble = scribble.transpose(2, 1, 0)[0]

    labeled_array, num_features = ndimage.label(scribble >= 255)
    centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features + 1))
    centers = np.array(centers)

    point_coords = torch.from_numpy(centers)
    point_coords = sam_predictor.transform.apply_coords_torch(point_coords, image.shape[:2])
    point_coords = point_coords.unsqueeze(0).to(device)
    point_labels = torch.from_numpy(np.array([1] * len(centers))).unsqueeze(0).to(device)

    if scribble_mode == "split":
        point_coords = point_coords.permute(1, 0, 2)
        point_labels = point_labels.permute(1, 0)

    masks, _, _ = sam_predictor.predict_torch(
        point_coords=point_coords if len(point_coords) > 0 else None,
        point_labels=point_labels if len(point_coords) > 0 else None,
        mask_input=None,
        boxes=None,
        multimask_output=False,
    )

    return masks


def process_other_tasks(
    transformed_image,
    task_type,
    image_pil,
    sam_predictor,
    text_prompt,
    box_threshold,
    text_threshold,
    iou_threshold,
):
    """
    Process other tasks (seg, inpainting, automatic) to generate masks.
    """
    boxes_filt, scores, pred_phrases = get_grounding_output(
        groundingdino_model, transformed_image, text_prompt, box_threshold, text_threshold
    )

    # Check if any boxes were filtered
    if boxes_filt is None or boxes_filt.size(0) == 0:
        print("No objects detected based on the prompt.")
        return None, None, None

    H, W = image_pil.size[1], image_pil.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()

    if task_type in ["seg", "inpainting", "automatic"]:
        sam_predictor.set_image(np.array(image_pil))

        if task_type == "automatic":
            print(f"Before NMS: {boxes_filt.shape[0]} boxes")
            nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]
            print(f"After NMS: {boxes_filt.shape[0]} boxes")

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image_pil.size[::-1]
        ).to(device)

        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

    return masks, boxes_filt, pred_phrases


def run_grounded_sam(input_file_path, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, iou_threshold, inpaint_mode, scribble_mode, openai_api_key, start_frame, stop_frame, skip_frame, color_palette):
    """
    Main function to run Grounded SAM tasks on an image or video.
    """
    if is_video_file(input_file_path):
        process_video(input_file_path, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, iou_threshold, inpaint_mode, scribble_mode, openai_api_key, start_frame, stop_frame, skip_frame, color_palette)
    else:
        # Load the image from the file path
        image_pil = Image.open(input_file_path).convert("RGB")
        input_image = {"image": image_pil, "mask": None}
        global blip_processor, blip_model, groundingdino_model, sam_predictor, sam_automask_generator, inpaint_pipeline

        # Initialize models if not already done
        sam_predictor, sam_automask_generator, groundingdino_model = initialize_models()

        size = image_pil.size

        # Get the base name for the output files
        basename = get_output_basename(input_image)

        # Ensure the directories exist
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualize'), exist_ok=True)

        # Save original image with the new basename
        image_path = os.path.join(output_dir, 'images', f"{basename}.jpg")
        image_pil.save(image_path)

        # Process the image based on task type
        masks, boxes_filt, pred_phrases = process_image(image_pil, None, task_type, sam_predictor, text_prompt, box_threshold, text_threshold, iou_threshold)

        # Detect classes and update classes.txt
        detected_classes = set(label.split('(')[0].strip() for label in pred_phrases)
        classes_file_path = os.path.join(output_dir, 'classes.txt')
        class_name_to_id = update_classes(detected_classes, classes_file_path)

        # Generate YOLOv8-seg annotations if applicable
        if task_type in ['seg', 'automatic']:
            index_list = [class_name_to_id[label.split('(')[0].strip()] for label in pred_phrases]
            convert_to_yolov8_seg_format(masks, index_list, output_dir, basename, size)

        # Visualize and save the result
        render_visualization(image_pil, masks, boxes_filt, pred_phrases, output_dir, basename, color_palette)

        # Return processed images
        if task_type == 'det':
            return render_detection(image_pil, boxes_filt, pred_phrases)
        elif task_type == 'automask':
            return render_automask(masks)
        elif task_type == 'scribble':
            return render_scribble(image_pil, masks, size)
        elif task_type == 'seg' or task_type == 'automatic':
            return render_segmentation(image_pil, masks, boxes_filt, pred_phrases, text_prompt)
        elif task_type == 'inpainting':
            return render_inpainting(image_pil, masks, inpaint_prompt, size)
        else:
            print(f"task_type: {task_type} error!")


def render_visualization(image_pil, masks, boxes_filt, pred_phrases, output_dir, basename, color_palette):
    """
    Render the visualization by overlaying segmentation masks and boxes on the original image.
    """
    # Convert PIL image to OpenCV format
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Load class name to ID mapping
    classes_file_path = os.path.join(output_dir, 'classes.txt')
    class_name_to_id = load_classes(classes_file_path)

    for mask, box, label in zip(masks, boxes_filt, pred_phrases):
        # Extract class ID from label
        class_name = label.split('(')[0].strip()
        class_id = class_name_to_id.get(class_name, -1)  # Default to -1 if class_name is not found

        if class_id == -1:
            continue  # Skip if class_id is not found

        # Use the class ID to determine color from the palette
        color_index = class_id % len(color_palette)  # Loop through the color palette
        color = color_palette[color_index]

        # Draw mask
        mask = mask[0].cpu().numpy()
        image[mask != 0] = image[mask != 0] * 0.5 + np.array(color) * 0.5

        # Draw bounding box
        box = box.numpy().astype(int)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, class_name, (box[0], box[1] - 10), font, 0.5, color, 2)

    # Save the visualized image
    visualize_path = os.path.join(output_dir, 'visualize', f"{basename}.jpg")
    cv2.imwrite(visualize_path, image)


def render_detection(image_pil, boxes_filt, pred_phrases):
    """
    Render detection results on the image.
    """
    image_draw = ImageDraw.Draw(image_pil)
    for box, label in zip(boxes_filt, pred_phrases):
        draw_box(box, image_draw, label)
    return [image_pil]


def render_automask(masks):
    """
    Render automask results.
    """
    full_img, res = show_anns(masks)
    return [full_img]


def render_scribble(image_pil, masks, size):
    """
    Render scribble task results on the image.
    """
    mask_image = Image.new("RGBA", size, color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)

    for mask in masks:
        draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

    image_pil = image_pil.convert("RGBA")
    image_pil.alpha_composite(mask_image)
    return [image_pil, mask_image]


def render_segmentation(image_pil, masks, boxes_filt, pred_phrases, text_prompt):
    """
    Render segmentation results on the image.
    """
    mask_image = Image.new("RGBA", image_pil.size, color=(0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)

    for mask in masks:
        draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

    image_draw = ImageDraw.Draw(image_pil)
    for box, label in zip(boxes_filt, pred_phrases):
        draw_box(box, image_draw, label)

    if text_prompt:
        image_draw.text((10, 10), text_prompt, fill="black")

    image_pil = image_pil.convert("RGBA")
    image_pil.alpha_composite(mask_image)
    return [image_pil, mask_image]


def render_inpainting(image_pil, masks, inpaint_prompt, size):
    """
    Render inpainting results on the image.
    """
    assert inpaint_prompt, "inpaint_prompt is not found!"
    if inpaint_mode == "merge":
        masks = torch.sum(masks, dim=0).unsqueeze(0)
        masks = torch.where(masks > 0, True, False)
    mask = masks[0][0].cpu().numpy()
    mask_pil = Image.fromarray(mask)

    if inpaint_pipeline is None:
        inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
        )
        inpaint_pipeline = inpaint_pipeline.to("cuda")

    image = inpaint_pipeline(
        prompt=inpaint_prompt,
        image=image_pil.resize((512, 512)),
        mask_image=mask_pil.resize((512, 512)),
    ).images[0]
    image = image.resize(size)

    return [image, mask_pil]


def convert_to_yolov8_seg_format(masks, index_list, output_dir, basename, image_size):
    """
    Convert segmentation masks to YOLOv8-seg format and save them to a file.
    """
    label_file_path = os.path.join(output_dir, "labels", f"{basename}.txt")
    with open(label_file_path, "w") as f:
        for mask, class_id in zip(masks, index_list):
            if class_id == -1:
                continue  # 不明なクラスは無視

            mask = mask.cpu().numpy().squeeze().astype(np.uint8)  # マスクをNumPy配列に変換
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # 輪郭が小さすぎるものは除外
                if len(contour) < 3:
                    continue

                # 輪郭座標を正規化し、フラットなリストに変換
                contour = contour.astype(float)
                contour[:, :, 0] /= image_size[0]  # x座標を画像幅で正規化
                contour[:, :, 1] /= image_size[1]  # y座標を画像高さで正規化
                contour_flat = contour.flatten().tolist()

                # クラスIDと座標を保存
                contour_str = " ".join(map(str, contour_flat))
                f.write(f"{class_id} {contour_str}\n")


def draw_mask(mask, draw, random_color=False):
    """
    Draw a mask on the image using the provided draw object.
    """
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)


def draw_box(box, draw, label=None):
    """
    Draw a bounding box with a label on the image.

    Args:
        box: The bounding box coordinates (x1, y1, x2, y2).
        draw: The ImageDraw object used to draw on the image.
        label: The label to be displayed with the box.
    """
    # Random color for the box
    color = tuple(np.random.randint(0, 255, size=3).tolist())

    # Draw the rectangle for the bounding box
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=2)

    # Draw the label if provided
    if label:
        font = ImageFont.load_default()
        # Calculate text bounding box
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        # Draw a rectangle for the text background
        draw.rectangle(bbox, fill=color)
        # Draw the text
        draw.text((box[0], box[1]), str(label), fill="white")


def get_output_basename(input_image):
    """
    Get the base name for the output files based on the input image name.
    If the name is not available, use a numeric postfix.
    """
    try:
        # Try to get the image name from the file path
        if "name" in input_image and input_image["name"]:
            # Extract the file name without extension
            basename = os.path.splitext(os.path.basename(input_image["name"]))[0]
        else:
            # If name is not available, use a numeric postfix
            counter = 0
            while True:
                basename = f"output_image_{counter}"
                output_path = os.path.join(output_dir, "images", f"{basename}.jpg")
                if not os.path.exists(output_path):
                    break
                counter += 1
    except Exception as e:
        print(f"Error getting image basename: {e}")
        basename = "output_image"

    return basename


def is_video_file(file_path):
    """
    Determine if the file is a video based on its extension.
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions


def process_video(input_video_path, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, iou_threshold, inpaint_mode, scribble_mode, openai_api_key, start_frame, stop_frame, skip_frame, color_palette):
    """
    Process a video file, frame by frame, using the specified task.
    """
    global blip_processor, blip_model, groundingdino_model, sam_predictor, sam_automask_generator, inpaint_pipeline

    # Initialize models if not already done
    sam_predictor, sam_automask_generator, groundingdino_model = initialize_models()

    # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)

    # Get the video filename without extension
    video_basename = os.path.splitext(os.path.basename(input_video_path))[0]

    # Create directories for output
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualize'), exist_ok=True)

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0

    # Determine stop frame
    if stop_frame < 0 or stop_frame >= total_frames:
        stop_frame = total_frames - 1

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret or frame_number > stop_frame:
            break  # Break if the video has ended or reached stop frame

        if frame_number < start_frame or (frame_number - start_frame) % skip_frame != 0:
            frame_number += 1
            continue

        # Convert frame to PIL Image
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        size = image_pil.size

        # Generate a unique basename for each frame
        basename = f"{video_basename}_frame_{frame_number}"

        # Process the frame based on task type
        masks, boxes_filt, pred_phrases = process_image(image_pil, None, task_type, sam_predictor, text_prompt, box_threshold, text_threshold, iou_threshold)

        # Check if pred_phrases is not None
        if pred_phrases is None:
            print(f"No predictions for frame {frame_number}. Skipping.")
            frame_number += 1
            continue

        # Detect classes and update classes.txt
        detected_classes = set(label.split('(')[0].strip() for label in pred_phrases)
        classes_file_path = os.path.join(output_dir, 'classes.txt')
        class_name_to_id = update_classes(detected_classes, classes_file_path)

        # Generate YOLOv8-seg annotations if applicable
        if task_type in ['seg', 'automatic']:
            index_list = [class_name_to_id[label.split('(')[0].strip()] for label in pred_phrases]
            convert_to_yolov8_seg_format(masks, index_list, output_dir, basename, size)

        # Visualize and save the result
        render_visualization(image_pil, masks, boxes_filt, pred_phrases, output_dir, basename, color_palette)

        # Save the original frame
        image_path = os.path.join(output_dir, 'images', f"{basename}.jpg")
        image_pil.save(image_path)

        frame_number += 1

    video_capture.release()


def process_file(input_file, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, iou_threshold, inpaint_mode, scribble_mode, openai_api_key, start_frame, stop_frame, skip_frame):
    """
    Determine if the input is a video or image, and process accordingly.
    """
    file_path = input_file.name

    # Load configuration including the color palette
    config = load_config(GRADIO_CONFIG_FILE)
    color_palette = config.get("color_palette", DEFAULT_COLOR_PALETTE)

    run_grounded_sam(file_path, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, iou_threshold, inpaint_mode, scribble_mode, openai_api_key, start_frame, stop_frame, skip_frame, color_palette)


def main():
    """
    Main function to set up and launch the Gradio interface.
    """
    parser = argparse.ArgumentParser("Grounded SAM demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    parser.add_argument("--port", type=int, default=7589, help="port to run the server")
    parser.add_argument("--no-gradio-queue", action="store_true", help="Disable Gradio queue")
    args = parser.parse_args()

    print(args)

    block = gr.Blocks()
    if not args.no_gradio_queue:
        block = block.queue()

    # Load previous configuration
    config = load_config(GRADIO_CONFIG_FILE)
    color_palette = config.get("color_palette", DEFAULT_COLOR_PALETTE)

    with block:
        with gr.Row():
            with gr.Column():
                input_file = gr.File(label="Upload Image or Video", file_count="single")
                task_type = gr.Dropdown(
                    ["scribble", "automask", "det", "seg", "inpainting", "automatic"],
                    value=config.get("task_type", "automatic"),
                    label="Task Type",
                )
                text_prompt = gr.Textbox(
                    label="Text Prompt", value=config.get("text_prompt", "default prompt")
                )
                inpaint_prompt = gr.Textbox(
                    label="Inpaint Prompt", value=config.get("inpaint_prompt", "")
                )
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    box_threshold = gr.Slider(
                        label="Box Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=config.get("box_threshold", 0.3),
                        step=0.05,
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=config.get("text_threshold", 0.25),
                        step=0.05,
                    )
                    iou_threshold = gr.Slider(
                        label="IOU Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=config.get("iou_threshold", 0.5),
                        step=0.05,
                    )
                    inpaint_mode = gr.Dropdown(
                        ["merge", "first"],
                        value=config.get("inpaint_mode", "merge"),
                        label="Inpaint Mode",
                    )
                    scribble_mode = gr.Dropdown(
                        ["merge", "split"],
                        value=config.get("scribble_mode", "split"),
                        label="Scribble Mode",
                    )
                    openai_api_key = gr.Textbox(
                        label="(Optional) OpenAI Key, Enable ChatGPT",
                        value=config.get("openai_api_key", ""),
                    )

            with gr.Accordion("Video Options", open=False):
                start_frame = gr.Number(
                    label="Start Frame", value=config.get("start_frame", 0), step=1
                )
                stop_frame = gr.Number(
                    label="Stop Frame", value=config.get("stop_frame", 1000), step=1
                )
                skip_frame = gr.Number(
                    label="Skip Frame", value=config.get("skip_frame", 1), step=1
                )

            with gr.Column():
                gallery = gr.Gallery(
                    label="Generated Images", show_label=False, elem_id="gallery"
                ).style(preview=True, grid=2, object_fit="scale-down")

        run_button.click(
            fn=process_file,
            inputs=[
                input_file,
                text_prompt,
                task_type,
                inpaint_prompt,
                box_threshold,
                text_threshold,
                iou_threshold,
                inpaint_mode,
                scribble_mode,
                openai_api_key,
                start_frame,
                stop_frame,
                skip_frame,
            ],
            outputs=gallery,
        )

    block.queue(concurrency_count=100)
    block.launch(server_name="0.0.0.0", server_port=args.port, debug=args.debug, share=args.share)

    # Save the current configuration
    current_config = {
        "task_type": task_type.value,
        "text_prompt": text_prompt.value,
        "inpaint_prompt": inpaint_prompt.value,
        "box_threshold": box_threshold.value,
        "text_threshold": text_threshold.value,
        "iou_threshold": iou_threshold.value,
        "inpaint_mode": inpaint_mode.value,
        "scribble_mode": scribble_mode.value,
        "openai_api_key": openai_api_key.value,
        "color_palette": color_palette,
    }
    save_config(current_config)


if __name__ == "__main__":
    main()
