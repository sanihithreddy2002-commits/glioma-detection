"""
Automatic Dataset Downloader from Kaggle
Downloads Brain MRI dataset automatically
"""

import os
import sys
import zipfile
from pathlib import Path

def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    print("Setting up Kaggle credentials...")
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("\n⚠️  Kaggle credentials not found!")
        print("\nTo download datasets, you need Kaggle API credentials:")
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. This downloads kaggle.json")
        print("5. Move kaggle.json to:", kaggle_dir)
        print("\nOr enter credentials manually:")
        
        username = input("Kaggle Username: ").strip()
        key = input("Kaggle API Key: ").strip()
        
        if username and key:
            with open(kaggle_json, 'w') as f:
                f.write(f'{{"username":"{username}","key":"{key}"}}')
            os.chmod(kaggle_json, 0o600)
            print("✓ Credentials saved!")
        else:
            print("❌ No credentials provided. Using sample data instead.")
            return False
    
    return True

def download_brain_mri_dataset():
    """Download Brain MRI dataset from Kaggle"""
    
    print("\n" + "="*60)
    print("DOWNLOADING BRAIN MRI DATASET FROM KAGGLE")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs("data/dataset", exist_ok=True)
    
    if not setup_kaggle_credentials():
        create_sample_dataset()
        return
    
    try:
        import kaggle
        
        # Popular Brain MRI datasets (choose one)
        datasets = [
            "masoudnickparvar/brain-tumor-mri-dataset",  # 7k images, well-labeled
            "sartajbhuvaji/brain-tumor-classification-mri",  # 3k images
            "ahmedhamada0/brain-tumor-detection",  # Good quality
        ]
        
        print("Available datasets:")
        for i, dataset in enumerate(datasets, 1):
            print(f"{i}. {dataset}")
        
        choice = input("\nSelect dataset (1-3) or press Enter for default [1]: ").strip()
        
        if choice == "":
            choice = "1"
        
        try:
            dataset_choice = datasets[int(choice) - 1]
        except:
            dataset_choice = datasets[0]
        
        print(f"\nDownloading: {dataset_choice}")
        print("This may take several minutes depending on your connection...\n")
        
        # Download
        kaggle.api.dataset_download_files(
            dataset_choice,
            path="data/dataset",
            unzip=True
        )
        
        print("\n✓ Dataset downloaded successfully!")
        print(f"Location: data/dataset/")
        
        # List downloaded files
        files = list(Path("data/dataset").rglob("*"))
        print(f"\nTotal files downloaded: {len(files)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nCreating sample dataset instead...")
        create_sample_dataset()
        return False

def create_sample_dataset():
    """Create sample dataset structure for testing"""
    print("\nCreating sample dataset structure...")
    
    dirs = [
        "data/dataset/Training/glioma",
        "data/dataset/Training/no_tumor",
        "data/dataset/Testing/glioma",
        "data/dataset/Testing/no_tumor",
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create a README
    readme = """# Sample Dataset Structure

Since Kaggle dataset wasn't downloaded, this is a sample structure.

To use real data:
1. Download Brain MRI dataset from Kaggle
2. Place images in the appropriate folders:
   - Training/glioma/
   - Training/no_tumor/
   - Testing/glioma/
   - Testing/no_tumor/

Or run: python download_dataset.py
"""
    
    with open("data/dataset/README.txt", "w") as f:
        f.write(readme)
    
    print("✓ Sample structure created!")
    print("\nYou can manually add images to:")
    for dir_path in dirs:
        print(f"  - {dir_path}")

def verify_dataset():
    """Verify dataset is downloaded correctly"""
    print("\n" + "="*60)
    print("VERIFYING DATASET")
    print("="*60 + "\n")
    
    dataset_path = Path("data/dataset")
    
    if not dataset_path.exists():
        print("❌ Dataset directory not found!")
        return False
    
    # Count files
    image_extensions = {'.jpg', '.jpeg', '.png', '.nii', '.nii.gz'}
    images = [f for f in dataset_path.rglob("*") if f.suffix.lower() in image_extensions]
    
    print(f"✓ Dataset directory exists")
    print(f"✓ Found {len(images)} medical images")
    
    if len(images) == 0:
        print("\n⚠️  No images found. Dataset may not be downloaded.")
        print("Run: python download_dataset.py")
        return False
    
    # Show sample structure
    print("\nDataset structure:")
    for item in sorted(dataset_path.iterdir())[:10]:
        if item.is_dir():
            count = len(list(item.rglob("*")))
            print(f"  📁 {item.name}/ ({count} files)")
        else:
            print(f"  📄 {item.name}")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("GLIOMA DETECTION SYSTEM - DATASET SETUP")
    print("="*60)
    
    download_brain_mri_dataset()
    verify_dataset()
    
    print("\n✓ Setup complete!")
    print("\nNext steps:")
    print("1. Review data in: data/dataset/")
    print("2. Run training: python train_model.py")
    print("3. Start server: python main.py")