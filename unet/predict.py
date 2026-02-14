import torch
import numpy as np
from PIL import Image
from model import UNET
from utils import load_checkpoint
import torchvision.transforms.functional as TF

def predict_single_image(img_path, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET(in_channels=3, out_channels=1).to(device)
    
    # Load the "extracted" weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load and preprocess image
    image = Image.open(img_path).convert("RGB")
    image = TF.to_tensor(TF.resize(image, (512, 512))).unsqueeze(0).to(device)

    with torch.no_grad():
        output = torch.sigmoid(model(image))
        prediction = (output > 0.5).float()
    
    # Save or show the result
    import matplotlib.pyplot as plt
    plt.imshow(prediction.cpu().squeeze(), cmap="gray")
    plt.show()

TEST_IMG_DIR = "unet/data/images/test/001002.png"

predict_single_image(TEST_IMG_DIR, "unet/model/best_model.pth.tar")