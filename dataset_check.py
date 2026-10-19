import os
import pandas as pd

base_dir = "MMRDR/MMRDR-OCT"
csv_path = os.path.join(base_dir, "OCT.csv")

if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
else:
    df = pd.read_csv(csv_path)
    print("--- Dataset Verification ---")
    print(f"Total records in CSV: {len(df)}")
    print("\nClass distribution (0: No DME, 1: NCI-DME, 2: CI-DME):")
    print(df['grade'].value_counts().sort_index())

    missing_images = 0
    for rel_path in df['image']:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            missing_images += 1

    print(f"\nVerification status: {len(df) - missing_images}/{len(df)} images found.")
    if missing_images == 0:
        print("Success! Dataset is completely ready for model training.")