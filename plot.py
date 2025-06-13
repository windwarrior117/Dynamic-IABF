#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting and visualization tool for DynamicIABF,IABF, NECST and UAE models.
This module provides visualization tools for comparing model performance.
All visualizations are in English.

This is the main entry point that maintains backward compatibility
while internally using the modular visual package.
"""

# 导入visual模块的功能
try:
    from visual.core import ModelVisualizer
    from visual.data_utils import extract_test_data as _extract_test_data
    from visual.model_utils import find_model_dirs as _find_model_dirs
    from visual.cli import main_cli
    
    # 模块化版本标志
    MODULAR_VERSION = True
    
except ImportError:
    print("Warning: Could not import visual module. Falling back to legacy code.")
    MODULAR_VERSION = False

# 保持向后兼容性的函数
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
    if MODULAR_VERSION:
        return _extract_test_data(dataset_name, num_samples, force_update)
    else:
        # 旧版功能将在这里实现，但现在只是一个占位符
        raise NotImplementedError("Legacy extract_test_data not implemented")

def extract_mnist_test_data(num_samples=10, force_update=False):
    """Legacy function for backward compatibility"""
    return extract_test_data("mnist", num_samples, force_update)

def find_model_dirs(base_dir="./results/", model_types=None):
    """
    Find model directories and generate appropriate model names
        
    Args:
        base_dir: Base directory for result files
        model_types: Model types to display
            
    Returns:
        tuple: (model_dirs, model_names)
    """
    if MODULAR_VERSION:
        return _find_model_dirs(base_dir, model_types)
    else:
        # 旧版功能将在这里实现，但现在只是一个占位符
        raise NotImplementedError("Legacy find_model_dirs not implemented")

# 当作为主程序运行时，调用命令行接口
if __name__ == "__main__":
    if MODULAR_VERSION:
        main_cli()
    else:
        # 旧版主函数将在这里实现，但现在只是一个占位符
        raise NotImplementedError("Legacy main function not implemented") 