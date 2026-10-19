import os
import argparse
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import timm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Target layer selector based on backbone architecture
def get_target_layer(model, model_name):
    if "convnext" in model_name:
        return [model.stages[-1].blocks[-1].norm]
    elif "densenet" in model_name:
        return [model.features.denseblock4.denselayer16.conv2]
    elif "resnet" in model_name:
        return [model.layer4[-1]]
    elif "efficientnet" in model_name:
        return [model.conv_head]
    elif "mobilenet" in model_name:
        return [model.conv_head]
    else:
        raise ValueError(f"Unsupported model for target layer auto-selection: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM Visualizations for DME OCT Scans")
    parser.add_argument("--model_name", type=str, default="convnext_tiny", 
                        choices=["efficientnet_b0", "resnet50", "densenet121", "mobilenetv3_large_100", "convnext_tiny"])
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples per class to visualize")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = f"saved_models/best_{args.model_name}.pth"
    output_dir = f"gradcam_results_{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    base_dir = "MMRDR/MMRDR-OCT"
    csv_path = os.path.join(base_dir, "OCT.csv")
    df = pd.read_csv(csv_path)

    # Load Model Checkpoint
    model = timm.create_model(args.model_name, pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    target_layers = get_target_layer(model, args.model_name)
    cam = GradCAM(model=model, target_layers=target_layers)

    # Preprocessing pipelines
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    class_names = {0: "No_DME", 1: "NCI_DME", 2: "CI_DME"}

    print(f"Generating Grad-CAM heatmaps for architecture: {args.model_name}...")

    for grade, class_label in class_names.items():
        sub_df = df[df['grade'] == grade].head(args.num_samples)
        
        for idx, row in sub_df.iterrows():
            rel_img_path = str(row['image'])
            img_path = os.path.join(base_dir, rel_img_path)

            raw_img = Image.open(img_path).convert('RGB').resize((224, 224))
            rgb_img = np.float32(raw_img) / 255.0

            input_tensor = preprocess(raw_img).unsqueeze(0).to(device)

            # Generate CAM mask
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Get model prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                pred_class = torch.argmax(outputs, dim=1).item()

            # Plot Side-by-Side Comparison
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(raw_img)
            axes[0].set_title(f"Original OCT B-Scan\nGround Truth: {class_label}")
            axes[0].axis("off")

            axes[1].imshow(visualization)
            axes[1].set_title(f"Grad-CAM Heatmap\nPredicted: {class_names[pred_class]}")
            axes[1].axis("off")

            save_file = os.path.join(output_dir, f"grade_{grade}_sample_{idx}_pred_{pred_class}.png")
            plt.tight_layout()
            plt.savefig(save_file, dpi=200)
            plt.close()

            print(f"Saved: {save_file}")

    print(f"\nAll Grad-CAM visualizations saved in '{output_dir}/'")

if __name__ == "__main__":
    main()