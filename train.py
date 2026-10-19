import os
import time
import argparse
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

# 1. Custom Dataset Class for MMRDR-OCT
class OCTDataset(Dataset):
    def __init__(self, df, base_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rel_img_path = str(self.df.loc[idx, 'image'])
        img_path = os.path.join(self.base_dir, rel_img_path)
        
        image = Image.open(img_path).convert('RGB')
        label = int(self.df.loc[idx, 'grade'])

        if self.transform:
            image = self.transform(image)

        return image, label

# 2. Training Epoch Loop
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()

# 3. Evaluation Loop
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    epoch_loss = running_loss / total
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, acc, macro_f1, all_labels, all_preds

# 4. Main Workflow
def main():
    parser = argparse.ArgumentParser(description="Train Pretrained Models on MMRDR-OCT with Weighted Loss")
    parser.add_argument("--model_name", type=str, default="convnext_tiny", 
                        choices=["efficientnet_b0", "resnet50", "densenet121", "mobilenetv3_large_100", "convnext_tiny"],
                        help="Model architecture from timm")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Architecture: {args.model_name}")

    base_dir = "MMRDR/MMRDR-OCT"
    csv_path = os.path.join(base_dir, "OCT.csv")

    # Data Augmentation and Preprocessing
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load and Split Data (80% Train, 20% Validation)
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['grade'])

    # Compute Inverse Class Weights for Imbalance Mitigation
    class_counts = train_df['grade'].value_counts().sort_index().values
    total_samples = len(train_df)
    num_classes = len(class_counts)
    weights = [total_samples / (num_classes * count) for count in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    print(f"Training Class Distribution: {dict(zip(range(num_classes), class_counts))}")
    print(f"Computed Class Penalty Weights: {class_weights.cpu().numpy().round(4)}")

    train_dataset = OCTDataset(train_df, base_dir, transform=data_transforms['train'])
    val_dataset = OCTDataset(val_df, base_dir, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load Pre-trained Model with 3 Output Classes
    model = timm.create_model(args.model_name, pretrained=True, num_classes=3)
    model = model.to(device)

    # Loss Function with Inverse Class Weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_macro_f1 = 0.0
    os.makedirs("saved_models", exist_ok=True)
    save_path = f"saved_models/best_{args.model_name}.pth"

    print("--- Starting Model Training ---")
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro-F1: {val_f1:.4f}")

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best checkpoint to {save_path}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s. Best Val Macro-F1: {best_macro_f1:.4f}")

    # Final Evaluation & Metrics Output
    model.load_state_dict(torch.load(save_path))
    _, final_acc, final_f1, targets, predictions = evaluate(model, val_loader, criterion, device)

    print("\n--- Final Classification Metrics ---")
    target_names = ['0: No DME', '1: NCI-DME', '2: CI-DME']
    print(classification_report(targets, predictions, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(targets, predictions))

if __name__ == "__main__":
    main()