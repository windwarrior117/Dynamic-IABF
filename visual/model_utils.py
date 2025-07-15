#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工具模块 - 提供模型识别和查找功能
"""

import os
import json

def identify_model_type(config):
    """
    从配置文件中识别模型类型
    
    Args:
        config: 模型配置字典
    
    Returns:
        str: 模型类型标识符
    """
    # 首先检查是否直接包含model_type字段
    if "model_type" in config:
        return config["model_type"]
        
    # 根据不同模型的特征来识别
    if "dynamic_noise" in config and config.get("dynamic_noise", False):
        return "DynamicIABF"
    elif "iabf" in config.get("model", "").lower():
        return "IABF"
    elif "necst" in config.get("model", "").lower():
        return "NECST"
    elif "uae" in config.get("model", "").lower():
        return "UAE"
    
    # 检查其他可能的字段
    if "noise_type" in config:
        if "dynamic" in config.get("noise_type", "").lower():
            return "DynamicIABF"
        return "IABF"
    
    # 如果无法确定，返回未知类型
    return "Unknown"

def find_model_dirs(base_dir="./results/", model_types=None):
    """
    增强版查找模型目录函数，支持新旧目录结构
    
    新目录结构按三个级别组织:
    1. 模型类型 (例如: DynamicIABF, IABF, NECST, UAE)
    2. 数据集名称 (例如: mnist, cifar10)
    3. 实验ID和日期 (例如: lr0.001_bs64_nh128_20250426_120000)
    
    Args:
        base_dir: 基础搜索目录
        model_types: 要查找的模型类型列表，例如 ["DynamicIABF", "IABF", "NECST", "UAE"]
    
    Returns:
        tuple: (model_dirs, model_names) - 模型目录和对应名称的列表
    """
    if model_types is None:
        model_types = ["DynamicIABF", "IABF", "NECST", "UAE"]
        
    model_dirs = []
    model_names = []
    
    # 新目录结构查找 - 按模型类型/数据集/实验ID组织
    for model_type in model_types:
        model_type_dir = os.path.join(base_dir, model_type)
        if os.path.exists(model_type_dir):
            # 遍历模型类型下的所有数据集目录
            for dataset_dir in os.listdir(model_type_dir):
                dataset_path = os.path.join(model_type_dir, dataset_dir)
                if os.path.isdir(dataset_path):
                    # 遍历数据集目录下的所有实验
                    for exp_dir in os.listdir(dataset_path):
                        exp_path = os.path.join(dataset_path, exp_dir)
                        if os.path.isdir(exp_path) and os.path.exists(os.path.join(exp_path, "config.json")):
                            rel_path = os.path.relpath(exp_path, base_dir)
                            
                            # 提取实验ID作为显示名称
                            # 新格式为: 实验ID_YYYYMMDD_HHMMSS
                            # 例如: lr0.001_bs64_nh128_20250426_120000
                            parts = exp_dir.split('_')
                            if len(parts) >= 2:  # 确保至少有实验ID和日期部分
                                # 提取日期时间部分并格式化
                                try:
                                    # 检查是否有日期部分（格式为YYYYMMDD）
                                    date_str = None
                                    for part in parts:
                                        if len(part) == 8 and part.isdigit():
                                            date_str = part
                                            break
                                    
                                    if date_str:
                                        # 找到日期部分的索引
                                        date_index = parts.index(date_str)
                                        # 日期之前的部分作为实验ID
                                        exp_id = "_".join(parts[:date_index])
                                        # 创建显示名称
                                        display_name = f"{model_type}-{dataset_dir}-{exp_id}"
                                    else:
                                        # 如果找不到日期部分，使用整个目录名称
                                        display_name = f"{model_type}-{dataset_dir}-{exp_dir}"
                                except:
                                    # 如果解析失败，使用完整路径
                                    display_name = f"{model_type}-{dataset_dir}-{exp_dir}"
                            else:
                                display_name = f"{model_type}-{dataset_dir}-{exp_dir}"
                            
                            model_dirs.append(rel_path)
                            model_names.append(display_name)
    
    # 如果未在新目录结构找到模型，尝试旧目录结构
    if len(model_dirs) == 0:
        print("未在新目录结构中找到模型，尝试兼容旧目录结构...")
        for root, dirs, files in os.walk(base_dir):
            if "config.json" in files:
                rel_path = os.path.relpath(root, base_dir)
                
                # 尝试从配置文件确定模型类型
                try:
                    with open(os.path.join(root, "config.json"), 'r') as f:
                        config = json.load(f)
                        model_type = identify_model_type(config)
                        
                        if model_type in model_types:
                            model_dirs.append(rel_path)
                            model_names.append(f"{model_type}-{os.path.basename(root)}")
                except:
                    continue
    
    if len(model_dirs) == 0:
        print("Warning: No model directories found in new or old directory structures")
    
    return model_dirs, model_names

def extract_noise_level(model_dir, base_dir="./results/"):
    """
    从配置文件中提取噪声水平
    
    Args:
        model_dir: 模型目录
        base_dir: 基础目录
        
    Returns:
        float: 噪声水平，如果未找到则返回None
    """
    config_path = os.path.join(base_dir, model_dir, "config.json")
    if not os.path.exists(config_path):
        return None
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # 对于DynamicIABF模型，使用test_noise或普通noise
        model_type = identify_model_type(config)
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
    except Exception as e:
        print(f"Error extracting noise level from {config_path}: {str(e)}")
    
    return None

def load_config(model_dir, base_dir="./results/"):
    """
    加载模型配置文件
    
    Args:
        model_dir: 模型目录
        base_dir: 基础目录
        
    Returns:
        dict: 配置字典，如果未找到则返回None
    """
    config_path = os.path.join(base_dir, model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} does not exist")
        return None
        
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {str(e)}")
        return None
