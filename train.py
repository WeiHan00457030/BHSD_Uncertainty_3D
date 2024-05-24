import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def evaluate_model(model, test_loader, device):
    model.eval()
    num_correct = 0
    num_pixels = 0
    total_dice = 0.0
    count = 0 
    
    with torch.no_grad(): 
        for images, masks in test_loader:

            images,masks = images.to(device),masks.to(device)
            
            outputs = model(images)
                                
            # Compute accuracy
            preds = torch.argmax(outputs, dim=1)
            mask_non_zero = masks > 0  # Create a mask for non-zero regions
            num_correct += (preds[mask_non_zero] == masks[mask_non_zero]).sum().item()
            num_pixels += mask_non_zero.sum().item()
            
            # Compute Dice and IoU for each class
            for cls in range(1, 2):  # Classes 1 to n
                pred_cls = (preds == cls)
                true_cls = (masks == cls)
                intersection = (pred_cls & true_cls).float().sum()
                total = pred_cls.float().sum() + true_cls.float().sum()

                if total > 0:
                    dice = (2. * intersection) / total
                    total_dice += dice
                    count += 1

    pixel_accuracy = num_correct / num_pixels
    avg_dice = total_dice / count if count > 0 else 0
    
    return pixel_accuracy, avg_dice

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=25):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
        
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

        for images, masks in progress_bar:

            images,masks = images.to(device),masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        # evaluate model
        test_acc,test_dice = evaluate_model(model,test_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Test_Acc:{test_acc:.4f}, Avg_Dice:{test_dice:.4f}")

    return model

