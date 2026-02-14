import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt
import os
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    save_best_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
from losses import DiceLoss

# HyperParameters
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOACHS = 25
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False 
IMAGE_DIR = "unet/data/images"
MASK_DIR = "unet/data/mask"

def plot_all_metrics(history):
    """
    Generates curves for Loss (BCE, Dice, Total), Accuracy, Precision, Recall, and Dice Score.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    os.makedirs("outputs", exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Loss Curves
    axs[0, 0].plot(epochs, history["train_loss"], label="Train Total Loss")
    axs[0, 0].plot(epochs, history["val_loss"], label="Val Total Loss")
    axs[0, 0].plot(epochs, history["val_bce"], '--', label="Val BCE Loss")
    axs[0, 0].plot(epochs, history["val_dice_loss"], '--', label="Val Dice Loss")
    axs[0, 0].set_title("Loss Function Components")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    # 2. Accuracy & Dice Score
    axs[0, 1].plot(epochs, history["val_accuracy"], label="Accuracy", color="green")
    axs[0, 1].plot(epochs, history["val_dice_score"], label="Dice Score", color="blue")
    axs[0, 1].set_title("Accuracy and Dice Score")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("Score")
    axs[0, 1].legend()

    # 3. Precision
    axs[1, 0].plot(epochs, history["val_precision"], label="Precision", color="purple")
    axs[1, 0].set_title("Precision Curve")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("Precision")
    axs[1, 0].legend()

    # 4. Recall
    axs[1, 1].plot(epochs, history["val_recall"], label="Recall", color="orange")
    axs[1, 1].set_title("Recall Curve")
    axs[1, 1].set_xlabel("Epochs")
    axs[1, 1].set_ylabel("Recall")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig("outputs/training_report.png")
    print("Full metrics report saved to outputs/training_report.png")
    plt.close()

def train_fn(loader, model, optimizer, bce_loss, dice_loss, scaler):
    loop = tqdm(loader)
    epoch_loss = 0
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.amp.autocast("cuda"):
            predictions = model(data)
            loss_bce = bce_loss(predictions, targets)
            loss_dice = dice_loss(predictions, targets)
            loss = loss_bce + loss_dice
        
        # backwards
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    return epoch_loss / len(loader)

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    optimizer = optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE,
        momentum=0.9,     
        weight_decay=1e-4 
    )

    train_loader, val_loader, test_loader = get_loaders(
        IMAGE_DIR,
        MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    history = {
        "train_loss": [], "val_loss": [], "val_bce": [],
        "val_dice_loss": [], "val_dice_score": [],
        "val_accuracy": [], "val_precision": [], "val_recall": []
    }

    best_dice = 0.0
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(NUM_EPOACHS):
        print(f"Epoch [{epoch+1}/{NUM_EPOACHS}]")
        
        # 1. Training Phase
        avg_train_loss = train_fn(train_loader, model, optimizer, bce_loss, dice_loss, scaler)
        history["train_loss"].append(avg_train_loss)

        # 2. Validation Phase (Ensure utils.check_accuracy returns a dict)
        metrics = check_accuracy(val_loader, model, bce_loss, dice_loss, device=DEVICE)
        
        history["val_loss"].append(metrics["loss"])
        history["val_bce"].append(metrics["bce"])
        history["val_dice_loss"].append(metrics["dice_loss"])
        history["val_dice_score"].append(metrics["dice_score"])
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_precision"].append(metrics["precision"])
        history["val_recall"].append(metrics["recall"])

        # 3. Checkpoint Logic
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        if metrics["dice_score"] > best_dice:
            best_dice = metrics["dice_score"]
            save_best_checkpoint(checkpoint, best_dice)

    # 4. Generate Final Learning Curves
    plot_all_metrics(history)

    # 5. Final Test Evaluation (One-time)
    plot_all_metrics(history)

    # LOAD THE BEST MODEL before testing
    print("Loading best checkpoint for final testing...")
    load_checkpoint(torch.load("best_checkpoint.pth.tar"), model)

    # Final Test Evaluation
    print("Final Evaluation on Test Set using BEST model:")
    test_metrics = check_accuracy(test_loader, model, bce_loss, dice_loss, device=DEVICE)
    
    # Print the "True" performance of your project
    print(f"Final Test Dice: {test_metrics['dice_score']:.4f}")
    print(f"Test Dice: {test_metrics['dice_score']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")

if __name__ == "__main__":
    main()