import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "MMRDR/MMRDR-OCT"
    csv_path = os.path.join(base_dir, "OCT.csv")
    df = pd.read_csv(csv_path)

    _, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['grade'])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = OCTDataset(val_df, base_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    models_info = [
        ("convnext_tiny", "ConvNeXt-Tiny"),
        ("densenet121", "DenseNet-121"),
        ("efficientnet_b0", "EfficientNet-B0"),
        ("resnet50", "ResNet-50"),
        ("mobilenetv3_large_100", "MobileNet-V3")
    ]

    results = []
    os.makedirs("evaluation_results", exist_ok=True)

    plt.figure(figsize=(10, 8))
    target_names = ['No DME', 'NCI-DME', 'CI-DME']

    for model_key, display_name in models_info:
        checkpoint_path = f"saved_models/best_{model_key}.pth"
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {display_name}: Checkpoint not found.")
            continue

        model = timm.create_model(model_key, pretrained=False, num_classes=3)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()

        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_probs.append(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_probs = np.vstack(all_probs)
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Calculate per-class F1
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        
        # One-vs-Rest ROC-AUC
        labels_bin = label_binarize(all_labels, classes=[0, 1, 2])
        auc_score = roc_auc_score(labels_bin, all_probs, multi_class='ovr', average='macro')

        results.append({
            'Model': display_name,
            'Accuracy': round(acc, 4),
            'Macro F1': round(macro_f1, 4),
            'ROC-AUC': round(auc_score, 4),
            'Class 0 F1': round(f1_per_class[0], 4),
            'Class 1 F1': round(f1_per_class[1], 4),
            'Class 2 F1': round(f1_per_class[2], 4),
        })

        # Save Confusion Matrix Plot
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Confusion Matrix: {display_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f"evaluation_results/cm_{model_key}.png", dpi=200)
        plt.close()

    summary_df = pd.DataFrame(results)
    summary_df.to_csv("evaluation_results/model_comparison_summary.csv", index=False)

    print("\n================ FINAL RESEARCH COMPARISON TABLE ================")
    print(summary_df.to_string(index=False))
    print("\nConfusion matrix plots and summary table saved to 'evaluation_results/'.")

if __name__ == "__main__":
    main()