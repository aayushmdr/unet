## Code for model utilities
import torch
import torch.nn as nn
import torchvision
from dataset import ECPC_Dataset
from torch.utils.data import DataLoader, Subset

def save_checkpoint(state, filename="unet/model/my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def save_best_checkpoint(checkpoint,ds):
            save_checkpoint(checkpoint, filename="unet/model/best_model.pth.tar")
            print(f"==> New Best Dice: {ds:.4f}, model saved!")

def get_loaders(
        img_dir,
        mask_dir,
        batch_size,
        train_transform,
        val_transform, 
        num_workers=4,
        pin_memory=True,
):
    # Create two instances of the dataset pointing to the same folder
    train_ds_base = ECPC_Dataset(img_dir, mask_dir, transform=train_transform)
    val_ds_base = ECPC_Dataset(img_dir, mask_dir, transform=val_transform)

    train_size = 477
    val_size = 477
    
    # Indices for the split
    indices = list(range(len(train_ds_base)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Use Subset to apply the correct transform to the correct indices
    train_dataset = Subset(train_ds_base, train_indices)
    val_dataset   = Subset(val_ds_base, val_indices) 
    test_dataset  = Subset(val_ds_base, test_indices) 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader, test_loader

def check_accuracy(loader, model, bce_loss, dice_loss, device="cuda"):
    model.eval()
    total_bce = 0
    total_dice_loss = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float().unsqueeze(1)
            
            raw_output = model(x)
            preds = torch.sigmoid(raw_output)
            preds_binary = (preds > 0.5).float()

            # Calculate Losses
            total_bce += bce_loss(raw_output, y).item()
            total_dice_loss += dice_loss(raw_output, y).item()

            # Calculate Pixel Counts for Metrics
            tp += ((preds_binary == 1) & (y == 1)).sum().item()
            fp += ((preds_binary == 1) & (y == 0)).sum().item()
            fn += ((preds_binary == 0) & (y == 1)).sum().item()
            tn += ((preds_binary == 0) & (y == 0)).sum().item()

    # Calculate metrics with epsilon to avoid division by zero
    e = 1e-7
    precision = tp / (tp + fp + e)
    recall = tp / (tp + fn + e)
    accuracy = (tp + tn) / (tp + tn + fp + fn + e)
    dice_score = (2 * tp) / (2 * tp + fp + fn + e)

    model.train()
    return {
        "loss": (total_bce / len(loader)) + (total_dice_loss / len(loader)),
        "bce": total_bce / len(loader),
        "dice_loss": total_dice_loss / len(loader),
        "dice_score": dice_score,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        torchvision.utils.save_image(
            preds, f"unet/{folder}/preds_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"unet/{folder}/gt_{idx}.png")

    model.train()
                                    

    


