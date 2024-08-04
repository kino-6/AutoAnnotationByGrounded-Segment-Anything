# Overview

* Automatic annotation tool using [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)

* Use gradio so you don't have to touch the code as much
* Annotation results support output for yolov8-seg

## Verification environment

* Windows11
  * CUDA 12.6
  * WSL2 (Ubuntu 22.04)
    * Python 3.10.12
    * `requirements.txt`

* Use PYENV as you wish. (We did not use pyenv in our test environment because we had concerns about its speed.)

## How to Use

1. Install [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
2. ```cd Grounded-Segment-Anything && python gradio_app.py --port (Arbitrary Ports)```
3. Enter the video data and the target class label you wish to automatically annotate in the prompt and execute
4. [formatted yolov8-seg data to outputs/](#output-gradio_apppy) images, labels, classes.txt(from prompt)
5. ```cd ../validate_yolo```
6. Run `main.ipynb`

### Output gradio_app.py

```log
├─images
├─labels
├─visualize
└─classes.txt
```
