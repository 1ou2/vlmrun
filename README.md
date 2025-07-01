# vlmrun
Run a vision language model

# EC2 drivers
```bash
sudo su ubuntu
cd
sudo apt update
nvidia-smi
sudo apt install nvidia-utils-550-server
lsmod | grep nvidia
sudo apt install nvidia-driver-550
sudo reboot
```


# Create venv
```
sudo apt install python3.12-venv 
python3 -m venv .venv
source .venv/bin activate
```


# Install the required packages for Qwen2.5-VL
pip install transformers
pip install torch torchvision torchaudio
pip install pillow requests
pip install qwen-vl-utils
pip install accelerate