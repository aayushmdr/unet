@echo off
python -m venv venv
venv/scripts/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python unet/predict.py
pause