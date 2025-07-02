from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class CelebASubset(Dataset):
    """CelebA dataset loader for the specific format you have"""
    def __init__(self, root_dir, attr_file, transform=None, max_samples=10000):
        self.root_dir = root_dir
        self.transform = transform
        
        # parse CelebA attributes file
        self.image_names, self.attributes = self._parse_attr_file(attr_file)
        
        # take subset for computational efficiency
        if max_samples < len(self.image_names):
            self.image_names = self.image_names[:max_samples]
            self.attributes = self.attributes[:max_samples]
        
        # extract protected attributes
        # from file: male is at index 20 (21st column), young is at index 39 (40th column)
        male_idx = 20  # 0-indexed position of 'Male' attribute
        young_idx = 39  # 0-indexed position of 'Young' attribute
        
        self.gender = (self.attributes[:, male_idx] == 1).astype(int)  # 1 for male, 0 for female
        self.age = (self.attributes[:, young_idx] == -1).astype(int)   # 1 for old, 0 for young
        

    def _parse_attr_file(self, attr_file):
        """Parse the CelebA attributes file format"""
        with open(attr_file, 'r') as f:
            lines = f.readlines()
        
        # First line is number of images
        num_images = int(lines[0].strip())
        
        # Second line is attribute names (we don't need to parse this)
        attr_names = lines[1].strip().split()
        
        # Parse image data
        image_names = []
        attributes = []
        
        for i in range(2, len(lines)):
            line = lines[i].strip()
            if not line: 
                continue
                
            parts = line.split()
            if len(parts) >= 41: 
                image_name = parts[0]
                attrs = [int(x) for x in parts[1:41]] 
                
                image_names.append(image_name)
                attributes.append(attrs)
        
        return image_names, np.array(attributes)
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.fromarray(np.zeros((218, 178, 3), dtype=np.uint8))
            if self.transform:
                image = self.transform(image)
            
        return image, self.gender[idx], self.age[idx]