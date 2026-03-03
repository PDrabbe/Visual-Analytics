# download_quickdraw.py
import urllib.request
import urllib.parse
import os
import numpy as np
from PIL import Image

def download_quickdraw_class(class_name, max_samples=1000, split_ratio=(0.7, 0.15, 0.15)):
    """Download one class from QuickDraw dataset."""
    
    # URL-encode class name to handle spaces
    encoded_name = urllib.parse.quote(class_name)
    base_url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{encoded_name}.npy"
    
    print(f"Downloading {class_name}...")
    
    # Download
    # Use sanitized name for local files (replace spaces with underscores)
    safe_name = class_name.replace(' ', '_')
    local_file = f"{safe_name}.npy"
    try:
        urllib.request.urlretrieve(base_url, local_file)
    except Exception as e:
        print(f"  Failed: {e}")
        return
    
    # Load
    data = np.load(local_file)
    data = data[:max_samples]  # Limit samples
    
    # Split into train/val/test
    n_train = int(len(data) * split_ratio[0])
    n_val = int(len(data) * split_ratio[1])
    
    splits = {
        'train': data[:n_train],
        'val': data[n_train:n_train+n_val],
        'test': data[n_train+n_val:]
    }
    
    # Save as images
    for split_name, split_data in splits.items():
        split_dir = f"data/quickdraw/{split_name}/{safe_name}"
        os.makedirs(split_dir, exist_ok=True)
        
        for i, img_data in enumerate(split_data):
            # Reshape from 784 to 28x28
            img = img_data.reshape(28, 28)
            
            # Convert to PIL and resize to 64x64
            img_pil = Image.fromarray(img).resize((64, 64))
            
            # Save
            img_pil.save(f"{split_dir}/{i:04d}.png")
    
    # Cleanup
    os.remove(local_file)
    print(f"  {class_name} downloaded and processed")

if __name__ == '__main__':
    # 29 diverse training classes spanning different shape categories
    classes = [
        # Animals (organic, curved shapes)
        'cat', 'dog', 'bird', 'fish', 'horse',
        # Food (round, blobby shapes)
        'apple', 'banana', 'cake', 'pizza',
        # Geometric / angular shapes
        'square', 'triangle', 'hexagon', 'diamond',
        # Vehicles (complex, structured)
        'car', 'bus', 'bicycle', 'truck',
        # Nature (varied shapes)
        'flower', 'cloud', 'lightning', 'mountain',
        # Objects (distinct features)
        'clock', 'key', 'scissors', 'eyeglasses',
        # Buildings / furniture (structural, angular)
        'door', 'table', 'chair', 'ladder',
    ]
    
    print(f"Downloading {len(classes)} QuickDraw classes...\n")
    
    for class_name in classes:
        download_quickdraw_class(class_name, max_samples=500)
    
    print(f"\nDownload complete! ({len(classes)} classes)")
    print("You can now run training:")
    print("  python main.py train --config config/config.yaml")
