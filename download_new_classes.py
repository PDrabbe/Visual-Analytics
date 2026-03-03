# download_new_classes.py
import urllib.request
import os
import numpy as np
from PIL import Image

def download_quickdraw_class(class_name, max_samples=100):
    """Download a QuickDraw class that wasn't in training."""
    
    # URL encode class name
    from urllib.parse import quote
    encoded_name = quote(class_name)
    base_url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{encoded_name}.npy"
    
    print(f"Downloading {class_name}...")
    
    # Download
    local_file = f"{class_name.replace(' ', '_')}.npy"
    try:
        urllib.request.urlretrieve(base_url, local_file)
    except Exception as e:
        print(f"  Failed: {e}")
        return False
    
    # Load
    data = np.load(local_file)
    data = data[:max_samples]
    
    # Save as images
    class_dir = f"custom_drawings/{class_name.replace(' ', '_')}"
    os.makedirs(class_dir, exist_ok=True)
    
    for i, img_data in enumerate(data):
        # Reshape from 784 to 28x28
        img = img_data.reshape(28, 28)
        
        # Resize to 64x64
        img_pil = Image.fromarray(img).resize((64, 64))
        
        # Save
        img_pil.save(f"{class_dir}/{i:04d}.png")
    
    # Cleanup
    os.remove(local_file)
    print(f"  Downloaded {len(data)} samples")
    return True

if __name__ == '__main__':
    # Download classes that were NOT in training
    new_classes = [
        'airplane',
        'book',
        'tree',
        'house',
        'umbrella',
        'guitar',
        'moon',
        'star',
    ]
    
    print("Downloading new QuickDraw classes (not in training)...\n")
    
    for class_name in new_classes:
        download_quickdraw_class(class_name, max_samples=50)
    
    print("\nNew classes downloaded to custom_drawings/")