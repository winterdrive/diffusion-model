"""
Dataset classes and data loading utilities for i-CLEVR diffusion model.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, List, Dict, Any


class iCLEVRDataset(Dataset):
    """
    i-CLEVR Dataset class with support for rectangular image sizes.
    
    The i-CLEVR dataset contains 18,009 images with size 320x240, 
    resolution 72dpi, and bit depth 32.
    """

    def __init__(
        self, 
        data_dir: str, 
        json_path: str, 
        object_path: str, 
        split: str = 'train', 
        img_size: Tuple[int, int] = (64, 64), 
        use_preset_size: bool = False
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Dataset directory
            json_path: Label JSON file path
            object_path: Objects JSON file path
            split: 'train', 'test', or 'new_test'
            img_size: Image size (width, height) or single integer
            use_preset_size: Whether to use preset resized images (skip resize)
        """
        self.data_dir = data_dir
        self.split = split
        self.use_preset_size = use_preset_size

        # Handle image size
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        # Load object dictionary
        with open(object_path, 'r') as f:
            self.object_dict = json.load(f)

        # Object names list
        self.obj_names = list(self.object_dict.keys())
        self.n_classes = len(self.obj_names)

        # Load data labels
        with open(json_path, 'r') as f:
            data = json.load(f)

        if split == 'train':
            # Training data: dict with filenames as keys and object lists as values
            self.data = data
            self.image_files = list(data.keys())
        else:
            # Test data: list of object lists
            self.data = data
            self.image_files = None  # Not applicable for test sets

        # Create name to index mapping
        self.name_to_idx = {name: idx for idx, name in enumerate(self.obj_names)}

        # Setup transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup image transformations."""
        transform_list = []
        
        if not self.use_preset_size:
            # Resize image if not using preset size
            transform_list.append(transforms.Resize(self.img_size))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        self.transform = transforms.Compose(transform_list)

    def _objects_to_multihot(self, obj_list: List[str]) -> torch.Tensor:
        """
        Convert object list to multi-hot encoding.
        
        Args:
            obj_list: List of object names
            
        Returns:
            torch.Tensor: Multi-hot encoded tensor
        """
        multihot = torch.zeros(self.n_classes)
        for obj in obj_list:
            if obj in self.name_to_idx:
                multihot[self.name_to_idx[obj]] = 1.0
        return multihot

    def __len__(self) -> int:
        """Return dataset length."""
        if self.split == 'train':
            return len(self.image_files)
        else:
            return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            tuple: (image_tensor, label_tensor)
        """
        if self.split == 'train':
            # Training mode: return image and its labels
            img_name = self.image_files[idx]
            obj_list = self.data[img_name]
            
            # Load image
            img_path = os.path.join(self.data_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            image = self.transform(image)
            
            # Convert objects to multi-hot encoding
            labels = self._objects_to_multihot(obj_list)
            
            return image, labels
        else:
            # Test mode: return only labels (for conditional generation)
            obj_list = self.data[idx]
            labels = self._objects_to_multihot(obj_list)
            
            # Return dummy image tensor and labels
            dummy_image = torch.zeros(3, *self.img_size)
            return dummy_image, labels


def create_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    img_size: Tuple[int, int] = (64, 64),
    num_workers: int = 4,
    use_preset_size: bool = False,
    train_json: str = 'train.json',
    test_json: str = 'test.json',
    new_test_json: str = 'new_test.json',
    objects_json: str = 'objects.json'
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training and testing.
    
    Args:
        data_dir: Dataset directory
        batch_size: Batch size for training
        img_size: Image size (width, height)
        num_workers: Number of worker processes
        use_preset_size: Whether to use preset resized images
        train_json: Training labels JSON filename
        test_json: Test labels JSON filename  
        new_test_json: New test labels JSON filename
        objects_json: Objects definition JSON filename
    
    Returns:
        dict: Dictionary containing data loaders
    """
    # Construct file paths
    train_json_path = os.path.join(data_dir, train_json)
    test_json_path = os.path.join(data_dir, test_json)
    new_test_json_path = os.path.join(data_dir, new_test_json)
    objects_json_path = os.path.join(data_dir, objects_json)
    
    # Create datasets
    datasets = {}
    
    if os.path.exists(train_json_path):
        datasets['train'] = iCLEVRDataset(
            data_dir=data_dir,
            json_path=train_json_path,
            object_path=objects_json_path,
            split='train',
            img_size=img_size,
            use_preset_size=use_preset_size
        )
    
    if os.path.exists(test_json_path):
        datasets['test'] = iCLEVRDataset(
            data_dir=data_dir,
            json_path=test_json_path,
            object_path=objects_json_path,
            split='test',
            img_size=img_size,
            use_preset_size=use_preset_size
        )
    
    if os.path.exists(new_test_json_path):
        datasets['new_test'] = iCLEVRDataset(
            data_dir=data_dir,
            json_path=new_test_json_path,
            object_path=objects_json_path,
            split='new_test',
            img_size=img_size,
            use_preset_size=use_preset_size
        )
    
    # Create data loaders
    data_loaders = {}
    
    for split, dataset in datasets.items():
        shuffle = (split == 'train')  # Only shuffle training data
        
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle  # Drop last incomplete batch only for training
        )
    
    return data_loaders


def get_object_names(data_dir: str, objects_json: str = 'objects.json') -> List[str]:
    """
    Get list of object names from objects JSON file.
    
    Args:
        data_dir: Dataset directory
        objects_json: Objects JSON filename
    
    Returns:
        list: List of object names
    """
    objects_path = os.path.join(data_dir, objects_json)
    
    with open(objects_path, 'r') as f:
        objects_dict = json.load(f)
    
    return list(objects_dict.keys())


def get_dataset_stats(data_dir: str, train_json: str = 'train.json') -> Dict[str, Any]:
    """
    Get dataset statistics.
    
    Args:
        data_dir: Dataset directory
        train_json: Training JSON filename
    
    Returns:
        dict: Dataset statistics
    """
    train_path = os.path.join(data_dir, train_json)
    
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    # Count object occurrences
    object_counts = {}
    total_objects = 0
    
    for img_name, obj_list in train_data.items():
        for obj in obj_list:
            object_counts[obj] = object_counts.get(obj, 0) + 1
            total_objects += 1
    
    # Calculate statistics
    num_images = len(train_data)
    avg_objects_per_image = total_objects / num_images
    
    stats = {
        'num_images': num_images,
        'num_unique_objects': len(object_counts),
        'total_object_instances': total_objects,
        'avg_objects_per_image': avg_objects_per_image,
        'object_counts': object_counts,
        'most_common_object': max(object_counts.items(), key=lambda x: x[1]),
        'least_common_object': min(object_counts.items(), key=lambda x: x[1])
    }
    
    return stats
