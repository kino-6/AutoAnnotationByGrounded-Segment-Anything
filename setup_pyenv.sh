#!/bin/bash

# 必要なパッケージをインストール
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    libffi-dev liblzma-dev git

# 古いpyenvディレクトリの削除
# rm -rf $(pwd)/.pyenv

# pyenvをカレントディレクトリにインストール
export PYENV_ROOT="$(pwd)/.pyenv"
curl https://pyenv.run | bash

# シェル設定ファイルにpyenvのパスを追加
echo "export PYENV_ROOT=\"$PYENV_ROOT\"" >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

# Pythonのバージョンを指定してインストール
PY_VERSION=3.10.12
pyenv install $PY_VERSION
pyenv global $PY_VERSION

# 仮想環境の作成
pyenv virtualenv $PY_VERSION venv

# 仮想環境をアクティブ化
source $PYENV_ROOT/versions/$PY_VERSION/envs/venv/bin/activate

# 仮想環境にパッケージをインストール
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 動作確認用スクリプトの実行
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('CUDA device name:', torch.cuda.get_device_name(torch.cuda.current_device()))
"
