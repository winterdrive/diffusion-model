#!/usr/bin/env python3
"""
Image preprocessing and visualization tools for i-CLEVR dataset.

This script provides utilities for testing different image transformation methods
and visualizing their effects on the dataset.
"""

import os
import sys
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
import random
import argparse

# Import evaluator
from evaluator import EvaluationModel


def create_transformation_methods():
    """
    Create various image transformation methods for comparison.

    Returns:
        dict: Mapping of transformation method names to transform functions
    """
    transform_methods = {}

    # Method 1: Direct resize to 64x64 (may distort aspect ratio)
    transform_methods['direct_resize'] = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Method 2: Preserve aspect ratio, resize then center crop
    transform_methods['resize_then_crop'] = transforms.Compose([
        transforms.Resize(85),  # Resize to slightly larger size, preserving aspect ratio
        transforms.CenterCrop(64),  # Center crop to fixed size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Method 3: Center crop first, then resize
    transform_methods['crop_then_resize'] = transforms.Compose([
        transforms.CenterCrop(min(320, 240)),  # Crop to square first
        transforms.Resize((64, 64)),  # Then resize to target size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Method 4: Resize short edge proportionally, then center crop
    transform_methods['aspect_resize_crop'] = transforms.Compose([
        transforms.Resize(64),  # Resize short edge to target size
        transforms.CenterCrop(64),  # Center crop to square
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Method 5: Pad with edge values to preserve original content
    transform_methods['pad_resize'] = transforms.Compose([
        transforms.Pad(padding=20, padding_mode='edge'),  # Add edge padding
        transforms.CenterCrop(min(320 + 40, 240 + 40)),  # Crop to square
        transforms.Resize((64, 64)),  # Resize to target size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return transform_methods


def visualize_transformation_methods(image_path, output_dir, methods=None):
    """
    Visualize effects of different transformation methods on the same image.

    Args:
        image_path (str): Path to input image
        output_dir (str): Output directory
        methods (dict, optional): Transformation methods dict, creates if not specified
    """
    if methods is None:
        methods = create_transformation_methods()
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Create figure
    plt.figure(figsize=(15, 6))
    
    # Show original image
    plt.subplot(1, len(methods) + 1, 1)
    plt.imshow(np.array(original_image))
    plt.title(f'Original\n({original_image.width}x{original_image.height})')
    plt.axis('off')
    
    # Apply and show each transformation method
    for i, (name, transform) in enumerate(methods.items()):
        # Apply transformation
        transformed = transform(original_image)
        
        # Denormalize for display
        img = transformed.permute(1, 2, 0).numpy() * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        
        # Show image
        plt.subplot(1, len(methods) + 1, i + 2)
        plt.imshow(img)
        plt.title(f'{name}\n({transformed.shape[1]}x{transformed.shape[2]})')
        plt.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comparison image
    img_name = os.path.basename(image_path).split('.')[0]
    plt.savefig(os.path.join(output_dir, f'{img_name}_comparison.png'), dpi=150)
    plt.close()


def create_preprocessed_dataset(data_dir, output_dir, method='direct_resize'):
    """
    Create preprocessed dataset with specified transformation method.
    
    Args:
        data_dir (str): Source data directory
        output_dir (str): Output directory for preprocessed images
        method (str): Transformation method name
    """
    # Get transformation method
    methods = create_transformation_methods()
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")
    
    transform = methods[method]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images
    images_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    output_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images with method: {method}")
    
    for img_file in tqdm(image_files, desc="Processing images"):
        # Load and transform image
        img_path = os.path.join(images_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformation (without normalization for saving)
        transform_for_save = transforms.Compose([
            step for step in transform.transforms[:-1]  # Exclude normalization
        ])
        
        transformed = transform_for_save(image)
        
        # Convert back to PIL and save
        if isinstance(transformed, torch.Tensor):
            transformed = transforms.ToPILImage()(transformed)
        
        output_path = os.path.join(output_images_dir, img_file)
        transformed.save(output_path)
    
    # Copy JSON files
    json_files = ['train.json', 'test.json', 'new_test.json', 'objects.json']
    for json_file in json_files:
        src_path = os.path.join(data_dir, json_file)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, json_file)
            import shutil
            shutil.copy2(src_path, dst_path)
    
    print(f"Preprocessed dataset saved to: {output_dir}")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="Image preprocessing for i-CLEVR dataset")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['visualize', 'create_dataset'],
                        help='Operation mode')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Input data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--method', type=str, default='direct_resize',
                        choices=['direct_resize', 'resize_then_crop', 'crop_then_resize', 
                               'aspect_resize_crop', 'pad_resize'],
                        help='Transformation method')
    parser.add_argument('--image_path', type=str,
                        help='Single image path for visualization mode')
    
    args = parser.parse_args()
    
    if args.mode == 'visualize':
        if not args.image_path:
            # Use first image from dataset
            images_dir = os.path.join(args.data_dir, 'images')
            if os.path.exists(images_dir):
                image_files = [f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    args.image_path = os.path.join(images_dir, image_files[0])
        
        if not args.image_path or not os.path.exists(args.image_path):
            raise ValueError("Valid image path required for visualization mode")
        
        print(f"Visualizing transformation methods on: {args.image_path}")
        visualize_transformation_methods(args.image_path, args.output_dir)
        print(f"Visualization saved to: {args.output_dir}")
        
    elif args.mode == 'create_dataset':
        print(f"Creating preprocessed dataset with method: {args.method}")
        create_preprocessed_dataset(args.data_dir, args.output_dir, args.method)


if __name__ == '__main__':
    main()
