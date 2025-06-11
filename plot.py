#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting and visualization tool for IABF, NECST and UAE models.
This module provides visualization tools for comparing model performance.
All visualizations are in English.
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.ticker import MaxNLocator
from glob import glob
import pandas as pd
import csv

def extract_test_data(dataset_name="mnist", num_samples=10, force_update=False):
    """
    Extract images from various test datasets and save to data directory
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', 'svhn', 'omniglot', 'celebA')
        num_samples: Number of samples to save
        force_update: Whether to update existing file forcefully
    
    Returns:
        bool: Whether data was successfully saved
    """
    # Target file path
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, f"{dataset_name}_test_data.pkl")
    
    # If file exists and not forcing update, return True
    if os.path.exists(output_file) and not force_update:
        print(f"Test data file already exists: {output_file}")
        return True
    
    try:
        test_samples = None
        
        # MNIST dataset
        if dataset_name.lower() == "mnist":
            from tensorflow.keras.datasets import mnist
            (_, _), (x_test, y_test) = mnist.load_data()
            
            # Normalize images to [0,1] range
            x_test = x_test.astype('float32') / 255.0
            
            # Reshape to vector form (28*28=784)
            x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
            
            # Extract samples
            test_samples = x_test_reshaped[:num_samples]
            
        # BinaryMNIST dataset
        elif dataset_name.lower() in ["binary_mnist", "binarymnist"]:
            # 尝试加载BinaryMNIST数据
            binary_mnist_dir = os.path.join("./data", "BinaryMNIST")
            test_file = os.path.join(binary_mnist_dir, "binarized_mnist_test.amat")
            
            if os.path.exists(test_file):
                # 从amat文件加载二进制MNIST数据
                x_test = np.loadtxt(test_file)
                
                # 已经是二值化的，范围在[0,1]之间
                x_test = x_test.astype('float32')
                
                # 重塑为28x28的图像
                # 数据已经是向量形式，无需重塑为向量
                
                # 提取样本
                test_samples = x_test[:num_samples]
            else:
                # 如果没有找到BinaryMNIST文件，尝试使用普通MNIST并二值化
                print(f"BinaryMNIST data not found in {binary_mnist_dir}, using MNIST and binarizing...")
                from tensorflow.keras.datasets import mnist
                (_, _), (x_test, y_test) = mnist.load_data()
                
                # 二值化到0和1
                x_test = (x_test > 127).astype('float32')
                
                # 重塑为向量形式 (28*28=784)
                x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
                
                # 提取样本
                test_samples = x_test_reshaped[:num_samples]
        
        # CIFAR10 dataset
        elif dataset_name.lower() == "cifar10":
            from tensorflow.keras.datasets import cifar10
            (_, _), (x_test, y_test) = cifar10.load_data()
            
            # Normalize images to [0,1] range
            x_test = x_test.astype('float32') / 255.0
            
            # Reshape to vector form (32*32*3=3072)
            x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
            
            # Extract samples
            test_samples = x_test_reshaped[:num_samples]
        
        # SVHN dataset
        elif dataset_name.lower() == "svhn":
            # Try to load from local directory first
            svhn_dir = os.path.join("./data", "svhn")
            local_file = os.path.join(svhn_dir, "test_32x32.mat")
            
            if os.path.exists(local_file):
                import scipy.io as sio
                test_data = sio.loadmat(local_file)
                x_test = test_data['X'].transpose(3, 0, 1, 2)  # Convert to [samples, height, width, channels]
            else:
                # Alternative: download from source
                print(f"SVHN data not found in {svhn_dir}, trying to download...")
                try:
                    import tensorflow_datasets as tfds
                    dataset = tfds.load('svhn_cropped', split='test')
                    x_test = []
                    for example in dataset.take(num_samples):
                        x_test.append(example['image'].numpy())
                    x_test = np.array(x_test)
                except:
                    raise ValueError(f"Failed to load SVHN data. Please download it to {svhn_dir} first.")
            
            # Normalize images to [0,1] range
            x_test = x_test.astype('float32') / 255.0
            
            # Reshape to vector form
            x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
            
            # Extract samples
            test_samples = x_test_reshaped[:num_samples]
        
        # Omniglot dataset
        elif dataset_name.lower() == "omniglot":
            # Try to load from local directory first
            omniglot_dir = os.path.join("./data", "omniglot")
            
            # Check if we have preprocessed data
            if os.path.exists(os.path.join(omniglot_dir, "processed/test_data.npy")):
                x_test = np.load(os.path.join(omniglot_dir, "processed/test_data.npy"))
                
                # Extract samples and reshape
                test_samples = x_test[:num_samples].reshape(min(num_samples, x_test.shape[0]), -1)
            else:
                # Try to load with tensorflow-datasets
                try:
                    import tensorflow_datasets as tfds
                    dataset = tfds.load('omniglot', split='test')
                    x_test = []
                    for example in dataset.take(num_samples):
                        # Convert to grayscale if needed and normalize
                        img = example['image'].numpy()
                        if len(img.shape) == 3 and img.shape[2] == 3:
                            # Convert RGB to grayscale if needed
                            img = np.mean(img, axis=2, keepdims=True)
                        img = img.astype('float32') / 255.0
                        x_test.append(img)
                    x_test = np.array(x_test)
                    
                    # Reshape to vector form
                    x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
                    
                    # Extract samples
                    test_samples = x_test_reshaped[:num_samples]
                except:
                    raise ValueError(f"Failed to load Omniglot data. Please download it to {omniglot_dir} first.")
        
        # CelebA dataset
        elif dataset_name.lower() == "celeba":
            # Try to load from local directory first
            celeba_dir = os.path.join("./data", "celebA")
            
            # Check if we have preprocessed data
            if os.path.exists(os.path.join(celeba_dir, "processed/test_data.npy")):
                x_test = np.load(os.path.join(celeba_dir, "processed/test_data.npy"))
                
                # Extract samples and reshape
                test_samples = x_test[:num_samples].reshape(min(num_samples, x_test.shape[0]), -1)
            else:
                # Try to load with tensorflow-datasets
                try:
                    import tensorflow_datasets as tfds
                    dataset = tfds.load('celeb_a', split='test')
                    x_test = []
                    for example in dataset.take(num_samples):
                        # Resize to standard size and normalize
                        img = example['image'].numpy()
                        # Resize if needed (e.g., to 64x64)
                        from skimage.transform import resize
                        img = resize(img, (64, 64, 3), anti_aliasing=True, preserve_range=True)
                        img = img.astype('float32') / 255.0
                        x_test.append(img)
                    x_test = np.array(x_test)
                    
                    # Reshape to vector form
                    x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
                    
                    # Extract samples
                    test_samples = x_test_reshaped[:num_samples]
                except:
                    raise ValueError(f"Failed to load CelebA data. Please download it to {celeba_dir} first.")
                    
        # Add support for other datasets as needed
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Save as pickle file
        if test_samples is not None:
            with open(output_file, 'wb') as f:
                pickle.dump(test_samples, f, pickle.HIGHEST_PROTOCOL)
            
            print(f"Saved {len(test_samples)} {dataset_name} test samples to: {output_file}")
            return True
        else:
            raise ValueError(f"Failed to extract test data for {dataset_name}")
    
    except Exception as e:
        print(f"Error extracting test data for {dataset_name}: {str(e)}")
        return False

# Keep the old function for backward compatibility
def extract_mnist_test_data(num_samples=10, force_update=False):
    """Legacy function for backward compatibility"""
    return extract_test_data("mnist", num_samples, force_update)

class ModelVisualizer:
    """Modular plotting tool for visualizing and comparing IABF, NECST and UAE model performance"""
    
    def __init__(self, base_dir="./results/", model_dirs=None, model_names=None, dataset_name="mnist"):
        """
        Initialize visualization tool
        
        Args:
            base_dir: Base directory for result files
            model_dirs: List of model result subdirectories, e.g. ["IABF/exp1", "NECST/exp1", "UAE/exp1"]
            model_names: Model names used in legends, e.g. ["IABF", "NECST", "UAE"]
            dataset_name: Name of dataset being visualized ('mnist', 'cifar10', 'svhn', 'omniglot', 'celebA')
        """
        self.base_dir = base_dir
        self.model_dirs = model_dirs if model_dirs else []
        self.model_names = model_names if model_names else ["Model " + str(i+1) for i in range(len(self.model_dirs))]
        self.dataset_name = dataset_name.lower()
        
        # Dataset-specific parameters
        self.dataset_params = {
            'mnist': {'img_shape': (28, 28), 'channels': 1, 'color_map': 'gray'},
            'binary_mnist': {'img_shape': (28, 28), 'channels': 1, 'color_map': 'gray'},
            'binarymnist': {'img_shape': (28, 28), 'channels': 1, 'color_map': 'gray'},
            'cifar10': {'img_shape': (32, 32), 'channels': 3, 'color_map': None},
            'svhn': {'img_shape': (32, 32), 'channels': 3, 'color_map': None},
            'omniglot': {'img_shape': (105, 105), 'channels': 1, 'color_map': 'gray'},  # Actual Omniglot size
            'celeba': {'img_shape': (64, 64), 'channels': 3, 'color_map': None}  # Resized CelebA
        }
        
        # Default to MNIST if dataset not in params
        if self.dataset_name not in self.dataset_params:
            print(f"Warning: Unknown dataset '{self.dataset_name}', defaulting to MNIST parameters")
            self.dataset_name = 'mnist'
            
        # Professional color scheme suitable for publications
        # Using colorblind-friendly colors that also work in grayscale
        self.colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442']
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>']
        self.line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
        
        # Default figure settings for publication quality
        self.dpi = 600  # High resolution for publication
        self.fig_width = 7.5  # Typical width for single-column journal figures (inches)
        self.fig_height = 5.5  # Height with good aspect ratio for publication
        
        # Ensure model names and directory counts match
        assert len(self.model_dirs) == len(self.model_names), "Model directory and name counts don't match"
        
        # Set publication-quality plot style
        self._set_publication_style()
        
        # Try to extract test data (if not already extracted)
        self._ensure_test_data_available()

    def _set_publication_style(self):
        """Set publication-quality matplotlib style settings"""
        # Use serif fonts for all text (standard in academic journals)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
        plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern for math text
        
        # Increase font sizes for readability
        plt.rcParams['font.size'] = 10        # Base font size
        plt.rcParams['axes.titlesize'] = 11   # Title font size
        plt.rcParams['axes.labelsize'] = 10   # Axis label font size
        plt.rcParams['xtick.labelsize'] = 9   # X-tick label font size
        plt.rcParams['ytick.labelsize'] = 9   # Y-tick label font size
        plt.rcParams['legend.fontsize'] = 9   # Legend font size
        
        # Line and marker settings
        plt.rcParams['lines.linewidth'] = 1.5     # Line width
        plt.rcParams['lines.markersize'] = 5      # Marker size
        plt.rcParams['axes.linewidth'] = 0.8      # Axis line width
        plt.rcParams['xtick.major.width'] = 0.8   # X-tick width
        plt.rcParams['ytick.major.width'] = 0.8   # Y-tick width
        plt.rcParams['xtick.minor.width'] = 0.6   # Minor X-tick width
        plt.rcParams['ytick.minor.width'] = 0.6   # Minor Y-tick width
        
        # Grid settings
        plt.rcParams['axes.grid'] = True          # Use grid
        plt.rcParams['grid.alpha'] = 0.3          # Grid transparency
        plt.rcParams['grid.linestyle'] = '--'     # Grid line style
        
        # Figure layout
        plt.rcParams['figure.constrained_layout.use'] = True  # Better automatic layout
        
        # Set the default DPI for saving figures
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['figure.dpi'] = 100  # Screen display DPI
        
        # Explicitly disable LaTeX - we don't need it and it causes errors if not installed
        plt.rcParams['text.usetex'] = False

    def _ensure_test_data_available(self):
        """Ensure test data file is available for current dataset"""
        extract_test_data(self.dataset_name)

    def _load_json(self, model_dir, filename="config.json"):
        """Load JSON configuration file"""
        filepath = os.path.join(self.base_dir, model_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} does not exist")
            return None
            
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _load_log(self, model_dir):
        """Extract training and validation losses from log file"""
        filepath = os.path.join(self.base_dir, model_dir, "log.txt")
        if not os.path.exists(filepath):
            print(f"Warning: Log file {filepath} does not exist")
            return None, None, None, None
        
        epochs = []
        train_losses = []
        valid_losses = []
        mi_values = []
        
        with open(filepath, 'r') as f:
            for line in f:
                if "Epoch" in line and "train loss" in line and "valid loss" in line:
                    parts = line.strip().split(',')
                    # Extract Epoch number
                    epoch = int(parts[0].split()[1])
                    epochs.append(epoch)
                    
                    # Extract training loss
                    for part in parts:
                        if "train loss" in part:
                            train_loss = float(part.split(':')[1].strip())
                            train_losses.append(train_loss)
                        elif "valid loss" in part and "time" not in part:
                            valid_loss = float(part.split(':')[1].strip())
                            valid_losses.append(valid_loss)
                        elif "mi" in part:
                            mi = float(part.split(':')[1].strip())
                            mi_values.append(mi)
                
                # Extract mutual information values from testing
                if "mutual information" in line:
                    parts = line.split()
                    mi_values.append(float(parts[-1]))
                    
        return epochs, train_losses, valid_losses, mi_values
    
    def _load_pickle(self, model_dir, filename="reconstr.pkl"):
        """Load reconstruction data"""
        filepath = os.path.join(self.base_dir, model_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: Pickle file {filepath} does not exist")
            return None
            
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def plot_learning_curves(self, save_path=None):
        """Plot learning curves (training and validation losses)"""
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        for i, model_dir in enumerate(self.model_dirs):
            epochs, train_losses, valid_losses, _ = self._load_log(model_dir)
            if epochs and train_losses and valid_losses:
                # Plot with consistent style per model
                ax.plot(epochs, train_losses, marker=self.markers[i], linestyle=self.line_styles[i], 
                       color=self.colors[i], label=f"{self.model_names[i]} - Training", 
                       markersize=6, markeredgewidth=1)
                
                ax.plot(epochs, valid_losses, marker=self.markers[i], linestyle='--',
                       color=self.colors[i], alpha=0.7, label=f"{self.model_names[i]} - Validation", 
                       markerfacecolor='none', markeredgewidth=1, markevery=2)
        
        # Set labels with proper font
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        
        # Use scientific notation for y-axis if appropriate
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
        
        # Better tick spacing
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Grid for readability
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add a box around the plot (publication standard)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        # Optimize legend appearance
        legend = ax.legend(frameon=True, framealpha=1, fancybox=False, edgecolor='black', loc='upper right')
        
        # Title positioned properly for publication
        ax.set_title('Training and Validation Loss Comparison')
        
        plt.tight_layout()
        
        if save_path:
            # Save as high-resolution vector and bitmap graphics
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1)
            
            # Also save as vector graphic format for publication
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
                print(f"Learning curves saved to {save_path} and {vector_path}")
            else:
                print(f"Learning curves saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_mutual_information(self, save_path=None):
        """Plot mutual information comparison"""
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        
        x = np.arange(len(self.model_dirs))
        bar_width = 0.6
        
        mi_values = []
        for model_dir in self.model_dirs:
            _, _, _, mi = self._load_log(model_dir)
            # Use the last mutual information value
            mi_values.append(mi[-1] if mi and len(mi) > 0 else 0)
        
        # Create bars with professional appearance
        bars = ax.bar(x, mi_values, bar_width, edgecolor='black', linewidth=1.2, zorder=3)
        
        # Color each bar according to the model color scheme
        for i, bar in enumerate(bars):
            bar.set_color(self.colors[i % len(self.colors)])
            bar.set_alpha(0.8)
        
        # Add value labels on the bars with professional formatting
        for i, v in enumerate(mi_values):
            if v > 0:  # Only add text if value is positive
                ax.text(i, v + max(mi_values) * 0.02, f"{v:.4f}", 
                       ha='center', va='bottom', fontsize=9, 
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Set descriptive labels
        ax.set_xlabel('Model')
        ax.set_ylabel('Mutual Information (nats)')
        
        # Customize x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names)
        
        # Add grid only on y-axis for clarity
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Add a box around the plot (publication standard)
        for spine in ax.spines.values():
            spine.set_visible(True)
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
        
        # Add a title
        ax.set_title('Mutual Information Comparison')
        
        plt.tight_layout()
        
        if save_path:
            # Save as high-resolution bitmap
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1)
            
            # Also save as vector format for publication
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
                print(f"Mutual information comparison saved to {save_path} and {vector_path}")
            else:
                print(f"Mutual information comparison saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_distribution_comparison(self, save_path=None):
        """Compare marginal distributions of models"""
        # Use wider figure for distribution plots
        fig_width = min(12, self.fig_width * 1.8)
        fig, axs = plt.subplots(1, len(self.model_dirs), figsize=(fig_width, self.fig_height), sharey=True)
        
        if len(self.model_dirs) == 1:
            axs = [axs]
        
        # Find global max frequency for consistent y-axis
        max_freq = 0
        all_distributions = []
        
        # First pass to collect all distributions and find max
        for i, model_dir in enumerate(self.model_dirs):
            dist_file = os.path.join(self.base_dir, model_dir, "distribution.pdf")
            if os.path.exists(dist_file.replace(".pdf", ".npy")):
                distribution = np.load(dist_file.replace(".pdf", ".npy"))
                all_distributions.append(distribution)
                hist, _ = np.histogram(distribution, bins=50)
                max_freq = max(max_freq, np.max(hist))
            else:
                all_distributions.append(None)
        
        # Second pass to plot with consistent styling
        for i, model_dir in enumerate(self.model_dirs):
            distribution = all_distributions[i]
            
            if distribution is not None:
                # Create histogram with professional styling
                n, bins, patches = axs[i].hist(distribution, bins=50, 
                                             edgecolor='black', linewidth=0.8, 
                                             color=self.colors[i], alpha=0.7, zorder=3)
                
                # Add KDE for smooth distribution visualization
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(distribution)
                    x_vals = np.linspace(min(distribution), max(distribution), 1000)
                    y_vals = kde(x_vals) * (max_freq / np.max(kde(x_vals)) if np.max(kde(x_vals)) > 0 else 1)
                    axs[i].plot(x_vals, y_vals, color='black', linewidth=1.5, zorder=4)
                except ImportError:
                    pass  # Skip KDE if scipy not available
                
                # Add distribution statistics
                mean = np.mean(distribution)
                std = np.std(distribution)
                axs[i].axvline(mean, color='red', linestyle='--', linewidth=1.5, zorder=5)
                axs[i].text(0.05, 0.95, f"Mean: {mean:.4f}\nStd: {std:.4f}", 
                          transform=axs[i].transAxes, fontsize=9,
                          verticalalignment='top', horizontalalignment='left',
                          bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
            else:
                # Show message if no data
                axs[i].text(0.5, 0.5, "No Distribution Data", 
                          fontsize=10, fontweight='bold',
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=axs[i].transAxes)
            
            # Set titles and labels
            axs[i].set_title(f"{self.model_names[i]}", fontsize=11)
            axs[i].set_xlabel("Activation Value")
            
            # Set consistent y-axis limits
            if max_freq > 0:
                axs[i].set_ylim(0, max_freq * 1.1)
            
            # Set professional grid
            axs[i].grid(True, linestyle='--', alpha=0.3)
            
            # Ensure box is visible
            for spine in axs[i].spines.values():
                spine.set_visible(True)
            
            if i == 0:
                axs[i].set_ylabel("Frequency")
        
        # Add a overall title
        fig.suptitle("Marginal Probability Distribution Analysis", fontsize=12, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            # Save high-resolution bitmap
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1)
            
            # Also save vector format
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
                print(f"Distribution comparison saved to {save_path} and {vector_path}")
            else:
                print(f"Distribution comparison saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_reconstruction_samples(self, num_samples=5, save_path=None):
        """Compare reconstruction samples between models with PSNR metric"""
        # Ensure num_samples is an integer and limit to prevent overcrowding
        num_samples = min(int(num_samples), 5)  # 限制最大样本数为5
        
        # 计算更合适的图形尺寸，增加空间
        fig_width = min(20, num_samples * 3)  # 增加宽度以适应更多的样本
        fig_height = min(16, len(self.model_dirs) * 3)  # 增加高度以适应更多的模型
        
        # 创建图形
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # 创建网格布局，增加更多间距
        gs = plt.GridSpec(len(self.model_dirs), num_samples*2, 
                        figure=fig, 
                        wspace=0.4,   # 增加水平间距
                        hspace=0.8,   # 增加垂直间距
                        left=0.05,    # 减小左边距，因为不再需要放置模型标签
                        right=0.95,
                        top=0.92,
                        bottom=0.15)  # 增加底部边距，为模型标签腾出空间
        
        # 创建轴对象
        axs = [[fig.add_subplot(gs[i, j]) for j in range(num_samples*2)] 
              for i in range(len(self.model_dirs))]
        
        # 加载统一测试数据
        dataset_test_file = os.path.join("./data", f"{self.dataset_name}_test_data.pkl")
        shared_test_data = None
        
        # 获取当前数据集的图像形状和颜色映射
        params = self.dataset_params.get(self.dataset_name, self.dataset_params['mnist'])
        img_shape = params['img_shape']
        color_map = params['color_map']
        channels = params['channels']
        
        # 如果是彩色图像，调整img_shape
        if channels == 3:
            img_shape = (*img_shape, 3)
        
        # 尝试加载共享测试数据
        if os.path.exists(dataset_test_file):
            try:
                with open(dataset_test_file, 'rb') as f:
                    shared_test_data = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load shared test data: {str(e)}")
        else:
            print(f"Warning: Cannot find shared test data: {dataset_test_file}")
        
        # Function to calculate PSNR
        def calculate_psnr(original, reconstructed):
            """Calculate Peak Signal-to-Noise Ratio"""
            if np.all(original == 0):  # Avoid division by zero
                return 0
            mse = np.mean((original - reconstructed) ** 2)
            if mse == 0:  # Perfect reconstruction
                return 100
            max_pixel = 1.0  # For normalized images
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            return psnr
        
        # Keep track of average PSNR per model
        psnr_values = []
        
        for i, model_dir in enumerate(self.model_dirs):
            # Load reconstruction data
            reconstr_data = self._load_pickle(model_dir)
            test_data = None

            # 尝试从模型目录加载测试数据
            test_data_path = os.path.join(self.base_dir, model_dir, "test_data.pkl")
            if os.path.exists(test_data_path):
                try:
                    with open(test_data_path, 'rb') as f:
                        test_data = pickle.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load model test data: {str(e)}")
            
            # 如果没有找到模型特定的测试数据，使用共享测试数据
            if test_data is None:
                test_data = shared_test_data
            
            model_psnr = []
            
            if reconstr_data is not None:
                # Determine image shape - 根据配置文件或重建数据推断
                config = self._load_json(model_dir)
                img_shape_from_config = None
                
                # 尝试从配置中获取数据集类型和图像形状
                if config and "dataset" in config:
                    dataset_type = config["dataset"].lower()
                    if dataset_type in self.dataset_params:
                        ds_params = self.dataset_params[dataset_type]
                        img_shape_from_config = ds_params['img_shape']
                        if ds_params['channels'] == 3:
                            img_shape_from_config = (*img_shape_from_config, 3)
                        color_map = ds_params['color_map']
                
                # 使用配置中的形状，如果可用
                if img_shape_from_config:
                    img_shape = img_shape_from_config
                
                # 如果配置中没有形状，则从重建数据推断
                elif len(reconstr_data.shape) == 2:
                    vector_size = reconstr_data.shape[1]
                    
                    # 推断可能的图像形状
                    if vector_size == 784:  # MNIST (28x28)
                        img_shape = (28, 28)
                        color_map = 'gray'
                    elif vector_size == 1024:  # 32x32 灰度
                        img_shape = (32, 32)
                        color_map = 'gray'
                    elif vector_size == 3072:  # CIFAR10/SVHN (32x32x3)
                        img_shape = (32, 32, 3)
                        color_map = None
                    elif vector_size == 4096:  # 64x64 灰度
                        img_shape = (64, 64)
                        color_map = 'gray'
                    elif vector_size == 12288:  # 64x64x3 彩色
                        img_shape = (64, 64, 3)
                        color_map = None
                    else:
                        # 尝试推断为平方形
                        side = int(np.sqrt(vector_size))
                        if side * side == vector_size:  # 完美平方
                            img_shape = (side, side)
                            color_map = 'gray'
                        else:
                            # 假设是RGB图像
                            side = int(np.sqrt(vector_size / 3))
                            if side * side * 3 == vector_size:
                                img_shape = (side, side, 3)
                                color_map = None
                            else:
                                print(f"Warning: Cannot determine image shape for vector size {vector_size}")
                                img_shape = (int(np.sqrt(vector_size)), int(np.sqrt(vector_size)))
                                color_map = 'gray'
                
                for j in range(num_samples):
                    if j < reconstr_data.shape[0]:
                        # Original image - if none, use zero matrix and show explanatory text
                        if test_data is not None and j < test_data.shape[0]:
                            img_original = test_data[j].reshape(img_shape)
                            has_original = True
                        else:
                            # Use zero matrix
                            img_original = np.zeros(img_shape)
                            has_original = False
                        
                        # Reconstructed image
                        img_reconstr = reconstr_data[j].reshape(img_shape)
                        
                        # Calculate PSNR if we have original data
                        psnr = 0
                        if has_original:
                            psnr = calculate_psnr(img_original, img_reconstr)
                            model_psnr.append(psnr)
                        
                        # Draw original image
                        ax_orig = axs[i][j*2]
                        if len(img_shape) == 3:  # Color image
                            ax_orig.imshow(img_original)
                        else:  # Grayscale image
                            ax_orig.imshow(img_original, cmap=color_map)
                        
                        # If no original data, add explicit annotation
                        if not has_original:
                            ax_orig.text(0.5, 0.5, "No Original Data", 
                                      color='red', fontsize=8, fontweight='bold',
                                      horizontalalignment='center',
                                      verticalalignment='center',
                                      transform=ax_orig.transAxes,
                                      bbox=dict(facecolor='white', alpha=0.8))
                        
                        # Set title with appropriate spacing
                        if j == 0:
                            ax_orig.set_title("Original", fontsize=10, pad=8)
                        ax_orig.axis('off')
                        
                        # Draw reconstructed image
                        ax_recon = axs[i][j*2+1]
                        if len(img_shape) == 3:  # Color image
                            ax_recon.imshow(img_reconstr)
                        else:  # Grayscale image
                            ax_recon.imshow(img_reconstr, cmap=color_map)
                        
                        # Add PSNR as subtitle to reconstruction if available - with better positioning
                        if has_original:
                            if j == 0:
                                ax_recon.set_title("Reconstructed\nPSNR: {:.2f} dB".format(psnr), 
                                                 fontsize=10, pad=8)
                            else:
                                ax_recon.set_title("PSNR: {:.2f} dB".format(psnr), 
                                                 fontsize=9, pad=8)
                        else:
                            if j == 0:
                                ax_recon.set_title("Reconstructed", fontsize=10, pad=8)
                        
                        ax_recon.axis('off')
            else:
                for j in range(num_samples*2):
                    axs[i][j].text(0.5, 0.5, "No Data", 
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 transform=axs[i][j].transAxes)
                    axs[i][j].axis('off')
            
            # 计算模型的平均PSNR
            model_label = self.model_names[i]
            if model_psnr:
                avg_psnr = np.mean(model_psnr)
                psnr_values.append(avg_psnr)
                model_label += f" (Avg PSNR: {avg_psnr:.2f} dB)"
            
            # 在每一行图像的下方添加模型标签，而不是左侧
            row_position = i / len(self.model_dirs)
            row_center = np.mean([axs[i][0].get_position().x0, axs[i][-1].get_position().x1])
            
            # 使用fig.text在行的下方添加模型标签
            # 对不同模型使用不同的垂直偏移
            if i == 0:  # IABF模型
                y_position = axs[i][0].get_position().y0 - 0.06
            elif i == 1:  # NECST模型
                y_position = axs[i][0].get_position().y0 - 0.09
            else:  # UAE模型及其他模型
                y_position = axs[i][0].get_position().y0 - 0.12  # 更大的偏移值，将UAE标签更往下移动
            
            # 在每行下方添加模型标签
            fig.text(row_center, y_position, 
                   model_label,
                   rotation=0,
                   horizontalalignment='center',
                   verticalalignment='top',
                   fontsize=11, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', 
                           boxstyle='round,pad=0.5'))
        
        # 添加总标题，调整位置
        dataset_title = self.dataset_name.upper()
        plt.suptitle(f"{dataset_title} Reconstruction Quality Comparison", fontsize=14, y=0.98)
        
        # 保存图像
        if save_path:
            # 保存高分辨率位图
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.3)
            
            # 同时保存矢量格式
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight', pad_inches=0.3)
                print(f"Reconstruction samples comparison saved to {save_path} and {vector_path}")
            else:
                print(f"Reconstruction samples comparison saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_test_metrics(self, save_path=None):
        """Plot test metrics comparison (test loss, mutual information, etc.)"""
        metrics = []
        for model_dir in self.model_dirs:
            log_file = os.path.join(self.base_dir, model_dir, "log.txt")
            if not os.path.exists(log_file):
                metrics.append({"loss": 0, "loss_per_pixel": 0, "mi": 0})
                continue
                
            test_loss = 0
            loss_per_pixel = 0
            mi = 0
            
            with open(log_file, 'r') as f:
                for line in f:
                    if "L2 test loss (per image)" in line:
                        test_loss = float(line.split(':')[1].strip())
                    elif "L2 test loss (per pixel)" in line:
                        loss_per_pixel = float(line.split(':')[1].strip())
                    elif "mutual information" in line:
                        mi = float(line.split()[-1].strip())
            
            metrics.append({"loss": test_loss, "loss_per_pixel": loss_per_pixel, "mi": mi})
        
        # Create subplots with common settings
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.fig_width * 1.4, self.fig_height))
        
        # Test loss comparison
        x = np.arange(len(self.model_dirs))
        width = 0.35
        
        loss_values = [m["loss"] for m in metrics]
        loss_per_pixel_values = [m["loss_per_pixel"] for m in metrics]
        
        # Plot bars with professional styling
        bars1 = ax1.bar(x - width/2, loss_values, width, 
                      edgecolor='black', linewidth=1, 
                      label='Loss per Image', zorder=3)
        
        bars2 = ax1.bar(x + width/2, loss_per_pixel_values, width, 
                      edgecolor='black', linewidth=1, 
                      label='Loss per Pixel', zorder=3)
        
        # Color bars according to model colors
        for i, bar in enumerate(bars1):
            bar.set_color(self.colors[i % len(self.colors)])
            bar.set_alpha(0.8)
        
        for i, bar in enumerate(bars2):
            bar.set_color(self.colors[i % len(self.colors)])
            bar.set_alpha(0.5)
        
        # Add value labels on bars
        for i, v in enumerate(loss_values):
            if v > 0:
                ax1.text(i - width/2, v + max(loss_values) * 0.02, f"{v:.4f}", 
                       ha='center', va='bottom', fontsize=8, 
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        for i, v in enumerate(loss_per_pixel_values):
            if v > 0:
                ax1.text(i + width/2, v + max(loss_per_pixel_values) * 0.02, f"{v:.4f}", 
                       ha='center', va='bottom', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Set labels
        ax1.set_ylabel('L2 Test Loss')
        ax1.set_title('Test Loss Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.model_names, rotation=30, ha='right')
        
        # Add professional legend
        ax1.legend(loc='upper right', frameon=True, framealpha=0.9, 
                 fancybox=False, edgecolor='black')
        
        # Add grid and box
        ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
        for spine in ax1.spines.values():
            spine.set_visible(True)
        
        # Set y-axis to start from 0
        ax1.set_ylim(bottom=0)
        
        # Mutual information comparison
        mi_values = [m["mi"] for m in metrics]
        bars3 = ax2.bar(x, mi_values, width*1.5, 
                      edgecolor='black', linewidth=1,
                      color='green', alpha=0.7, zorder=3)
        
        # Add value labels
        for i, v in enumerate(mi_values):
            if v > 0:
                ax2.text(i, v + max(mi_values) * 0.02 if max(mi_values) > 0 else 0.01, 
                       f"{v:.4f}", ha='center', va='bottom', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        ax2.set_ylabel('Mutual Information (nats)')
        ax2.set_title('Mutual Information Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.model_names, rotation=30, ha='right')
        
        # Add grid and box
        ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
        for spine in ax2.spines.values():
            spine.set_visible(True)
        
        # Set y-axis to start from 0
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            # Save high-resolution bitmap
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1)
            
            # Also save vector format
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
                print(f"Test metrics comparison saved to {save_path} and {vector_path}")
            else:
                print(f"Test metrics comparison saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def collect_reconstruction_errors(self, csv_save_path=None):
        """
        Collect reconstruction error metrics from all models and save to CSV
        
        Args:
            csv_save_path: Path to save CSV file
            
        Returns:
            pandas.DataFrame: DataFrame with reconstruction error metrics
        """
        data = []
        
        for i, model_dir in enumerate(self.model_dirs):
            model_name = self.model_names[i]
            log_file = os.path.join(self.base_dir, model_dir, "log.txt")
            
            # Default values
            metrics = {
                "Model": model_name,
                "Noise Level": 0,
                "Loss (per image)": 0,
                "Loss (per pixel)": 0,
            }
            
            # Extract noise level from config
            config = self._load_json(model_dir)
            if config and "noise" in config:
                metrics["Noise Level"] = float(config["noise"])
                
            # Extract metrics from log file
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    for line in f:
                        if "L2 test loss (per image)" in line:
                            metrics["Loss (per image)"] = float(line.split(':')[1].strip())
                        elif "L2 test loss (per pixel)" in line:
                            metrics["Loss (per pixel)"] = float(line.split(':')[1].strip())
            
            data.append(metrics)
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV if path provided
        if csv_save_path:
            os.makedirs(os.path.dirname(os.path.abspath(csv_save_path)), exist_ok=True)
            df.to_csv(csv_save_path, index=False)
            print(f"Reconstruction errors saved to {csv_save_path}")
            
        return df
    
    def plot_reconstruction_errors_table(self, save_path=None, csv_path=None):
        """
        Create a visual table of reconstruction error metrics
        
        Args:
            save_path: Path to save the table image
            csv_path: Path to save or load CSV data (if None, data will be collected but not saved to CSV)
            
        Returns:
            matplotlib.figure.Figure: Figure object with the table
        """
        # Collect data
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            csv_save_path = csv_path if csv_path else None
            df = self.collect_reconstruction_errors(csv_save_path)
            
        # Format the data for better visualization
        df_formatted = df.copy()
        for col in df.columns:
            if col != "Model":
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                
        # Create figure and axis - size depends on rows
        fig_height = max(2.5, min(8, 1.0 + len(df) * 0.5))
        fig, ax = plt.subplots(figsize=(self.fig_width, fig_height))
        
        # Remove axis
        ax.axis('off')
        ax.axis('tight')
        
        # Create table with professional styling
        table = ax.table(
            cellText=df_formatted.values,
            colLabels=df.columns,
            loc='center',
            cellLoc='center',
            edges='closed',
            bbox=[0, 0, 1, 1]
        )
        
        # Style the table for publication quality
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Professional color scheme
        header_color = '#4472C4'
        alt_row_color = '#F2F2F2'
        model_col_color = '#E6EFF9'
        border_color = '#BFBFBF'
        
        # Style header row and cells
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(border_color)
            
            if row == 0:  # Header row
                cell.set_facecolor(header_color)
                cell.set_text_props(color='white', fontweight='bold')
                cell.set_height(0.15)
            else:  # Data rows
                if col == 0:  # Model column
                    cell.set_facecolor(model_col_color)
                elif row % 2 == 0:  # Even rows
                    cell.set_facecolor(alt_row_color)
        
        # Add a professional title
        plt.suptitle('Reconstruction Error Metrics', fontsize=12, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            # Save high-resolution bitmap
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1)
            
            # Also save vector format
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
                print(f"Reconstruction errors table saved to {save_path} and {vector_path}")
            else:
                print(f"Reconstruction errors table saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        return fig
    
    def plot_noise_analysis(self, save_path=None):
        """Analyze model performance at different noise levels"""
        # Collect data points
        noise_levels = []
        test_losses = []
        mi_values = []
        model_types = []
        model_names = []
        
        for i, model_dir in enumerate(self.model_dirs):
            # Extract noise level
            noise = self._extract_noise_level(model_dir)
            if noise is None:
                continue
                
            # Extract test loss and mutual information
            log_file = os.path.join(self.base_dir, model_dir, "log.txt")
            if not os.path.exists(log_file):
                continue
                
            test_loss = 0
            mi = 0
            
            with open(log_file, 'r') as f:
                for line in f:
                    if "L2 test loss (per image)" in line:
                        try:
                            test_loss = float(line.split(':')[1].strip())
                        except:
                            test_loss = 0
                    elif "mutual information" in line:
                        try:
                            mi = float(line.split()[-1].strip())
                        except:
                            mi = 0
            
            # 获取配置和模型类型
            config = self._load_json(model_dir)
            model_type = _identify_model_type(config) if config else "Unknown"
            
            # 收集数据
            noise_levels.append(noise)
            test_losses.append(test_loss)
            mi_values.append(mi)
            model_types.append(model_type)
            model_names.append(self.model_names[i])
        
        # Exit if not enough data
        if len(noise_levels) < 1:
            print("警告: 需要至少一个带噪声级别的模型才能进行比较")
            plt.figure(figsize=(self.fig_width, self.fig_height))
            plt.text(0.5, 0.5, "没有足够的噪声数据进行分析", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                print(f"Saved empty noise analysis to {save_path}")
            else:
                plt.show()
            plt.close()
            return
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(self.fig_width * 1.5, self.fig_height))
        
        # 确保axes是一个列表
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # 创建标记和线型的循环
        markers = ['o', 's', '^', 'D', 'v']
        
        # 分组数据并按模型类型绘制测试损失
        unique_model_types = list(set(model_types))
        
        # 绘制测试损失与噪声关系图
        ax1 = axes[0]
        for i, model_type in enumerate(unique_model_types):
            indices = [j for j, m in enumerate(model_types) if m == model_type]
            if indices:
                x = [noise_levels[j] for j in indices]
                y = [test_losses[j] for j in indices]
                labels = [model_names[j] for j in indices]
                
                # 按噪声水平排序以便正确连接线条
                sorted_data = sorted(zip(x, y, labels))
                sorted_x, sorted_y, sorted_labels = zip(*sorted_data) if sorted_data else ([], [], [])
                
                # 绘制散点
                ax1.scatter(sorted_x, sorted_y, 
                          marker=markers[i % len(markers)],
                          color=self.colors[i % len(self.colors)],
                          s=80, label=model_type, zorder=10)
                
                # 如果有多个点，则连线
                if len(sorted_x) > 1:
                    ax1.plot(sorted_x, sorted_y, 
                           linestyle=self.line_styles[i % len(self.line_styles)],
                           color=self.colors[i % len(self.colors)],
                           linewidth=1.5, alpha=0.7)
                
                # 添加数据点标签
                for j, (x_val, y_val, label) in enumerate(zip(sorted_x, sorted_y, sorted_labels)):
                    ax1.annotate(label, (x_val, y_val), 
                               textcoords="offset points",
                               xytext=(0, 10), 
                               ha='center', fontsize=8)
        
        # 设置测试损失图表样式
        ax1.set_xlabel('噪声水平')
        ax1.set_ylabel('L2 测试损失')
        ax1.set_title('测试损失 vs. 噪声水平')
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # 确保坐标轴有合理的范围
        if len(noise_levels) > 0:
            ax1.set_xlim(min(noise_levels) * 0.9, max(noise_levels) * 1.1)
        if len(test_losses) > 0 and max(test_losses) > 0:
            ax1.set_ylim(0, max(test_losses) * 1.2)
        
        # 添加图例
        if len(unique_model_types) > 1:
            ax1.legend(loc='best', frameon=True, framealpha=0.9, 
                     fancybox=False, edgecolor='black')
        
        # 显示边框
        for spine in ax1.spines.values():
            spine.set_visible(True)
        
        # 绘制互信息与噪声关系图
        ax2 = axes[1]
        for i, model_type in enumerate(unique_model_types):
            indices = [j for j, m in enumerate(model_types) if m == model_type]
            if indices:
                x = [noise_levels[j] for j in indices]
                y = [mi_values[j] for j in indices]
                labels = [model_names[j] for j in indices]
                
                # 按噪声水平排序以便正确连接线条
                sorted_data = sorted(zip(x, y, labels))
                sorted_x, sorted_y, sorted_labels = zip(*sorted_data) if sorted_data else ([], [], [])
                
                # 绘制散点
                ax2.scatter(sorted_x, sorted_y, 
                          marker=markers[i % len(markers)],
                          color=self.colors[i % len(self.colors)],
                          s=80, label=model_type, zorder=10)
                
                # 如果有多个点，则连线
                if len(sorted_x) > 1:
                    ax2.plot(sorted_x, sorted_y, 
                           linestyle=self.line_styles[i % len(self.line_styles)],
                           color=self.colors[i % len(self.colors)],
                           linewidth=1.5, alpha=0.7)
                
                # 添加数据点标签
                for j, (x_val, y_val, label) in enumerate(zip(sorted_x, sorted_y, sorted_labels)):
                    ax2.annotate(label, (x_val, y_val), 
                               textcoords="offset points",
                               xytext=(0, 10), 
                               ha='center', fontsize=8)
        
        # 设置互信息图表样式
        ax2.set_xlabel('噪声水平')
        ax2.set_ylabel('互信息 (nats)')
        ax2.set_title('互信息 vs. 噪声水平')
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # 确保坐标轴有合理的范围
        if len(noise_levels) > 0:
            ax2.set_xlim(min(noise_levels) * 0.9, max(noise_levels) * 1.1)
        if len(mi_values) > 0 and max(mi_values) > 0:
            ax2.set_ylim(0, max(mi_values) * 1.2)
        
        # 如果需要添加图例
        if len(unique_model_types) > 1 and not ax1.get_legend():
            ax2.legend(loc='best', frameon=True, framealpha=0.9, 
                     fancybox=False, edgecolor='black')
        
        # 添加总标题
        plt.suptitle('不同噪声水平对模型性能的影响', fontsize=12, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved noise analysis to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_dynamic_iabf_features(self, save_path=None):
        """特别针对DynamicIABF模型的特性可视化"""
        
        # 收集所有DynamicIABF模型的数据
        dynamic_models = []
        
        for i, model_dir in enumerate(self.model_dirs):
            config = self._load_json(model_dir)
            model_type = _identify_model_type(config) if config else "Unknown"
            
            if model_type == "DynamicIABF" and config:
                # 基本模型参数 - 确保所有参数都有默认值
                is_adaptive = config.get("adaptive_noise", False)
                is_progressive = config.get("progressive_training", False)
                min_noise = float(config.get("noise_min", 0.01))
                max_noise = float(config.get("noise_max", 0.3))
                adapt_rate = float(config.get("noise_adapt_rate", 0.05))
                
                # 从日志中提取性能指标
                log_file = os.path.join(self.base_dir, model_dir, "log.txt")
                test_loss = 0
                mi = 0
                
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        for line in f:
                            if "L2 test loss (per image)" in line:
                                try:
                                    test_loss = float(line.split(':')[1].strip())
                                except:
                                    test_loss = 0
                            elif "mutual information" in line:
                                try:
                                    mi = float(line.split()[-1].strip())
                                except:
                                    mi = 0
                
                # 收集模型数据
                dynamic_models.append({
                    "dir": model_dir,
                    "name": os.path.basename(model_dir),
                    "adaptive": is_adaptive,
                    "progressive": is_progressive,
                    "min_noise": min_noise,
                    "max_noise": max_noise,
                    "adapt_rate": adapt_rate,
                    "test_noise": float(config.get("test_noise", config.get("noise", 0.1))),
                    "test_loss": test_loss,
                    "mi": mi,
                    "color": self.colors[i % len(self.colors)]
                })
        
        # 如果没有DynamicIABF模型，返回
        if not dynamic_models:
            print("No DynamicIABF models found for feature visualization")
            return
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(self.fig_width * 1.4, self.fig_height))
        
        # 按特性分组模型
        adaptive_only = [m for m in dynamic_models if m['adaptive'] and not m['progressive']]
        progressive_only = [m for m in dynamic_models if not m['adaptive'] and m['progressive']]
        both_features = [m for m in dynamic_models if m['adaptive'] and m['progressive']]
        standard = [m for m in dynamic_models if not m['adaptive'] and not m['progressive']]
        
        # 绘制按特性分组的测试损失
        ax1 = axes[0]
        
        # 准备数据
        labels = []
        losses = []
        colors = []
        
        # 添加所有模型数据确保图表不为空
        for i, model in enumerate(dynamic_models):
            labels.append(model['name'])
            losses.append(model['test_loss'])
            
            # 根据类型使用不同颜色
            if model['adaptive'] and model['progressive']:
                colors.append(self.colors[2])
            elif model['adaptive']:
                colors.append(self.colors[0])
            elif model['progressive']:
                colors.append(self.colors[1])
            else:
                colors.append(self.colors[3])
        
        # 绘制条形图
        if labels and losses:
            bars = ax1.bar(range(len(labels)), losses, color=colors)
            
            # 设置图表样式
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.set_ylabel('L2 测试损失')
            ax1.set_title('不同DynamicIABF配置的测试损失')
            
            # 添加数值标签
            for i, v in enumerate(losses):
                ax1.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=8)
            
            # 添加图例
            import matplotlib.patches as mpatches
            legends = []
            if adaptive_only:
                legends.append(mpatches.Patch(color=self.colors[0], label='仅自适应噪声'))
            if progressive_only:
                legends.append(mpatches.Patch(color=self.colors[1], label='仅渐进式训练'))
            if both_features:
                legends.append(mpatches.Patch(color=self.colors[2], label='自适应+渐进式'))
            if standard:
                legends.append(mpatches.Patch(color=self.colors[3], label='标准DynamicIABF'))
                
            if legends:
                ax1.legend(handles=legends, loc='best')
        
        # 绘制噪声范围与测试噪声的关系
        ax2 = axes[1]
        
        # 清空坐标轴，确保即使数据不足也能有基本显示
        ax2.clear()
        
        # 为每个模型准备数据点
        if dynamic_models:
            for i, model in enumerate(dynamic_models):
                # 设置颜色
                if model['adaptive'] and model['progressive']:
                    color = self.colors[2]  # 同时使用两种特性
                    label = "自适应+渐进式"
                elif model['adaptive']:
                    color = self.colors[0]  # 仅自适应
                    label = "仅自适应噪声"
                elif model['progressive']:
                    color = self.colors[1]  # 仅渐进式
                    label = "仅渐进式训练"
                else:
                    color = self.colors[3]  # 其他情况
                    label = "标准DynamicIABF"
                
                # 绘制噪声范围
                ax2.plot([i, i], [model['min_noise'], model['max_noise']], 
                        color=color, linewidth=5, alpha=0.7,
                        label=label if i==0 else "")
                
                # 标记测试噪声水平
                ax2.scatter([i], [model['test_noise']], color='red', s=100, zorder=10,
                           label="测试噪声水平" if i==0 else "")
                
                # 显示自适应调整率
                if model['adaptive']:
                    ax2.text(i, model['max_noise'], f"调整率: {model['adapt_rate']}", 
                            ha='center', va='bottom', fontsize=8)
        
            # 设置图表样式
            ax2.set_xticks(range(len(dynamic_models)))
            ax2.set_xticklabels([m['name'] for m in dynamic_models], rotation=45, ha='right')
            ax2.set_ylabel('噪声水平')
            ax2.set_title('DynamicIABF噪声范围与测试噪声')
            
            # 确保y轴范围合理
            if len(dynamic_models) > 0:
                min_y = min(model['min_noise'] for model in dynamic_models) * 0.9
                max_y = max(model['max_noise'] for model in dynamic_models) * 1.1
                ax2.set_ylim(min_y, max_y)
            
            # 添加图例
            handles, labels = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax2.legend(by_label.values(), by_label.keys(), loc='best')
        else:
            # 如果没有数据，显示提示
            ax2.text(0.5, 0.5, "没有DynamicIABF模型数据", 
                    ha='center', va='center', fontsize=12, 
                    transform=ax2.transAxes)
        
        # 添加总标题
        plt.suptitle('DynamicIABF模型特性分析', fontsize=12, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved DynamicIABF feature visualization to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_all(self, output_dir="./plots/"):
        """生成所有支持的图表"""
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Generating plots in {output_dir}...")
        
        # 绘制学习曲线
        self.plot_learning_curves(os.path.join(output_dir, 'learning_curves.png'))
        
        # 绘制互信息比较
        self.plot_mutual_information(os.path.join(output_dir, 'mutual_information.png'))
        
        # 绘制分布比较
        self.plot_distribution_comparison(os.path.join(output_dir, 'distribution_comparison.png'))
        
        # 绘制重建样本
        self.plot_reconstruction_samples(save_path=os.path.join(output_dir, 'reconstruction_samples.png'))
        
        # 绘制测试指标
        self.plot_test_metrics(os.path.join(output_dir, 'test_metrics.png'))
        
        # 绘制噪声分析
        self.plot_noise_analysis(os.path.join(output_dir, 'noise_analysis.png'))
        
        # 生成重建误差表格
        self.plot_reconstruction_errors_table(save_path=os.path.join(output_dir, 'reconstruction_errors_table.png'))
        
        # 检查是否有DynamicIABF模型，如果有则绘制特性分析
        has_dynamic_model = False
        for model_dir in self.model_dirs:
            config = self._load_json(model_dir)
            if config and _identify_model_type(config) == "DynamicIABF":
                has_dynamic_model = True
                break
        
        if has_dynamic_model:
            self.plot_dynamic_iabf_features(os.path.join(output_dir, 'dynamic_iabf_features.png'))
        
        print(f"All plots generated successfully in {output_dir}")

    def _extract_noise_level(self, model_dir):
        """Extract noise level from configuration file"""
        config = self._load_json(model_dir)
        if config:
            # 对于DynamicIABF模型，使用test_noise或普通noise
            model_type = _identify_model_type(config)
            if model_type == "DynamicIABF":
                if "test_noise" in config:
                    return float(config["test_noise"])
                elif "noise" in config:
                    return float(config["noise"])
                else:
                    # 如果没有明确指定，则使用最大噪声和最小噪声的平均值
                    min_noise = float(config.get("noise_min", 0.01))
                    max_noise = float(config.get("noise_max", 0.3))
                    return (min_noise + max_noise) / 2
            # 对于普通模型，使用noise
            elif "noise" in config:
                return float(config["noise"])
        return None

def _identify_model_type(config):
    """
    Identify model type from configuration
    
    Returns:
        str: "DynamicIABF", "IABF", "NECST", "UAE" or "Unknown"
    """
    if not config:
        return "Unknown"
        
    # 首先检查是否为DynamicIABF
    if ("use_dynamic_model" in config and config["use_dynamic_model"] == True) or \
       ("adaptive_noise" in config and config["adaptive_noise"] == True) or \
       ("progressive_training" in config and config["progressive_training"] == True):
        return "DynamicIABF"
        
    # 检查是否为IABF
    if "miw" in config and float(config["miw"]) > 0 and "flip_samples" in config and int(config["flip_samples"]) > 0:
        return "IABF"
    
    # 检查是否为NECST
    if ("miw" not in config or float(config.get("miw", 0)) == 0) and \
       ("flip_samples" not in config or int(config.get("flip_samples", 0)) == 0) and \
       "noise" in config and float(config["noise"]) > 0:
        return "NECST"
    
    # 检查是否为UAE
    if ("miw" not in config or float(config.get("miw", 0)) == 0) and \
       ("flip_samples" not in config or int(config.get("flip_samples", 0)) == 0) and \
       ("noise" not in config or float(config.get("noise", 0)) == 0):
        return "UAE"
    
    return "Unknown"

def find_model_dirs(base_dir="./results/", model_types=None):
    """
    Find model directories by type
    
    Args:
        base_dir: Base directory to search
        model_types: List of model types to find, e.g. ["DynamicIABF", "IABF", "NECST", "UAE"]
    
    Returns:
        tuple: (model_dirs, model_names) - lists of model directories and corresponding names
    """
    if model_types is None:
        model_types = ["DynamicIABF", "IABF", "NECST", "UAE"]
        
    model_dirs = []
    model_names = []
    
    # Walk through directory structure
    for root, dirs, files in os.walk(base_dir):
        if "config.json" in files:
            # Found a model directory with config.json
            rel_path = os.path.relpath(root, base_dir)
            
            # Load config to determine model type
            with open(os.path.join(root, "config.json"), 'r') as f:
                try:
                    config = json.load(f)
                    model_type = _identify_model_type(config)
                    
                    if model_type in model_types:
                        model_dirs.append(rel_path)
                        model_names.append(f"{model_type}-{os.path.basename(root)}")
                except:
                    continue
    
    return model_dirs, model_names

def main():
    """Main function, handle command line arguments and perform visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Performance Visualization Tool")
    parser.add_argument("--base_dir", type=str, default="./results/", 
                        help="Base directory for result files")
    parser.add_argument("--model_dirs", type=str, nargs="+", 
                        help="List of model result subdirectories")
    parser.add_argument("--model_names", type=str, nargs="+", 
                        help="Model names used in legends")
    parser.add_argument("--model_types", type=str, nargs="+", 
                        help="Model types to display, e.g. DynamicIABF, IABF, NECST, UAE")
    parser.add_argument("--output_dir", type=str, default="./plots/", 
                        help="Output directory for saving images")
    parser.add_argument("--auto_discover", action="store_true", 
                        help="Automatically discover model directories")
    parser.add_argument("--plot_type", type=str, 
                        choices=["all", "learning", "mi", "distribution", "reconstruction", 
                                "metrics", "noise", "table"], 
                        default="all", help="Type of plot to generate")
    parser.add_argument("--extract_data", action="store_true",
                        help="Force extraction and update of test data")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of test samples to extract")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "binary_mnist", "binarymnist", "cifar10", "svhn", "omniglot", "celeba"],
                        help="Dataset type for visualization")
    
    args = parser.parse_args()
    
    # If data extraction specified, extract test data first
    if args.extract_data:
        extract_test_data(args.dataset, num_samples=args.num_samples, force_update=True)
    else:
        # Ensure test data exists
        extract_test_data(args.dataset, num_samples=args.num_samples)
    
    if args.auto_discover:
        # If model types specified, only show those types
        selected_types = args.model_types if args.model_types else ["DynamicIABF", "IABF", "NECST", "UAE"]
        model_dirs, model_names = find_model_dirs(args.base_dir, selected_types)
        if not model_dirs:
            print("No model result directories found")
            return
        print(f"Found models: {', '.join(model_names)}")
    else:
        model_dirs = args.model_dirs if args.model_dirs else []
        model_names = args.model_names if args.model_names else ["Model " + str(i+1) for i in range(len(model_dirs))]
    
    # Create visualizer with specified dataset
    visualizer = ModelVisualizer(args.base_dir, model_dirs, model_names, args.dataset)
    
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate plot based on specified type
    if args.plot_type == "all":
        try:
            visualizer.plot_all(args.output_dir)
        except Exception as e:
            print(f"Error generating some plots: {str(e)}")
    elif args.plot_type == "learning":
        save_path = os.path.join(args.output_dir, "learning_curves.png")
        visualizer.plot_learning_curves(save_path)
    elif args.plot_type == "mi":
        save_path = os.path.join(args.output_dir, "mutual_information.png")
        visualizer.plot_mutual_information(save_path)
    elif args.plot_type == "distribution":
        save_path = os.path.join(args.output_dir, "distribution_comparison.png")
        visualizer.plot_distribution_comparison(save_path)
    elif args.plot_type == "reconstruction":
        save_path = os.path.join(args.output_dir, "reconstruction_samples.png")
        visualizer.plot_reconstruction_samples(save_path=save_path)
    elif args.plot_type == "metrics":
        save_path = os.path.join(args.output_dir, "test_metrics.png")
        visualizer.plot_test_metrics(save_path)
    elif args.plot_type == "noise":
        save_path = os.path.join(args.output_dir, "noise_analysis.png")
        visualizer.plot_noise_analysis(save_path)
    elif args.plot_type == "table":
        csv_path = os.path.join(args.output_dir, "reconstruction_errors.csv")
        save_path = os.path.join(args.output_dir, "reconstruction_errors_table.png")
        visualizer.plot_reconstruction_errors_table(save_path, csv_path)

if __name__ == "__main__":
    main() 