#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据工具模块 - 测试数据提取和处理功能
"""

import os
import pickle
import numpy as np

def extract_test_data(dataset_name="mnist", num_samples=10, force_update=False):
    """
    从各种测试数据集提取图像并保存到数据目录
    
    Args:
        dataset_name: 数据集名称 ('mnist', 'cifar10', 'svhn', 'omniglot', 'celebA')
        num_samples: 要保存的样本数量
        force_update: 是否强制更新现有文件
    
    Returns:
        bool: 数据是否成功保存
    """
    # 目标文件路径
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    output_file = os.path.join(data_dir, f"{dataset_name}_test_data.pkl")
    
    # 如果文件存在且不强制更新，返回True
    if os.path.exists(output_file) and not force_update:
        print(f"Test data file already exists: {output_file}")
        return True
    
    try:
        test_samples = None
        
        # MNIST数据集
        if dataset_name.lower() == "mnist":
            from tensorflow.keras.datasets import mnist
            (_, _), (x_test, y_test) = mnist.load_data()
            
            # 将图像规范化到[0,1]范围
            x_test = x_test.astype('float32') / 255.0
            
            # 重塑为向量形式 (28*28=784)
            x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
            
            # 提取样本
            test_samples = x_test_reshaped[:num_samples]
            
        # BinaryMNIST数据集
        elif dataset_name.lower() in ["binary_mnist", "binarymnist"]:
            # 尝试加载BinaryMNIST数据
            binary_mnist_dir = os.path.join("./data", "BinaryMNIST")
            test_file = os.path.join(binary_mnist_dir, "binarized_mnist_test.amat")
            
            if os.path.exists(test_file):
                # 从amat文件加载二进制MNIST数据
                x_test = np.loadtxt(test_file)
                
                # 已经是二值化的，范围在[0,1]之间
                x_test = x_test.astype('float32')
                
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
        
        # CIFAR10数据集
        elif dataset_name.lower() == "cifar10":
            from tensorflow.keras.datasets import cifar10
            (_, _), (x_test, y_test) = cifar10.load_data()
            
            # 将图像规范化到[0,1]范围
            x_test = x_test.astype('float32') / 255.0
            
            # 重塑为向量形式 (32*32*3=3072)
            x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
            
            # 提取样本
            test_samples = x_test_reshaped[:num_samples]
        
        # SVHN数据集
        elif dataset_name.lower() == "svhn":
            # 首先尝试从本地目录加载
            svhn_dir = os.path.join("./data", "svhn")
            local_file = os.path.join(svhn_dir, "test_32x32.mat")
            
            if os.path.exists(local_file):
                import scipy.io as sio
                test_data = sio.loadmat(local_file)
                x_test = test_data['X'].transpose(3, 0, 1, 2)  # 转换为[samples, height, width, channels]
            else:
                # 替代方案：从源下载
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
            
            # 将图像规范化到[0,1]范围
            x_test = x_test.astype('float32') / 255.0
            
            # 重塑为向量形式
            x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
            
            # 提取样本
            test_samples = x_test_reshaped[:num_samples]
        
        # Omniglot数据集
        elif dataset_name.lower() == "omniglot":
            # 首先尝试从本地目录加载
            omniglot_dir = os.path.join("./data", "omniglot")
            
            # 检查是否有预处理数据
            if os.path.exists(os.path.join(omniglot_dir, "processed/test_data.npy")):
                x_test = np.load(os.path.join(omniglot_dir, "processed/test_data.npy"))
                
                # 提取样本并重塑
                test_samples = x_test[:num_samples].reshape(min(num_samples, x_test.shape[0]), -1)
            else:
                # 尝试使用tensorflow-datasets加载
                try:
                    import tensorflow_datasets as tfds
                    dataset = tfds.load('omniglot', split='test')
                    x_test = []
                    for example in dataset.take(num_samples):
                        # 如需将图像转换为灰度并规范化
                        img = example['image'].numpy()
                        if len(img.shape) == 3 and img.shape[2] == 3:
                            # 如需将RGB转换为灰度
                            img = np.mean(img, axis=2, keepdims=True)
                        img = img.astype('float32') / 255.0
                        x_test.append(img)
                    x_test = np.array(x_test)
                    
                    # 重塑为向量形式
                    x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
                    
                    # 提取样本
                    test_samples = x_test_reshaped[:num_samples]
                except:
                    raise ValueError(f"Failed to load Omniglot data. Please download it to {omniglot_dir} first.")
        
        # CelebA数据集
        elif dataset_name.lower() == "celeba":
            # 首先尝试从本地目录加载
            celeba_dir = os.path.join("./data", "celebA")
            
            # 检查是否有预处理数据
            if os.path.exists(os.path.join(celeba_dir, "processed/test_data.npy")):
                x_test = np.load(os.path.join(celeba_dir, "processed/test_data.npy"))
                
                # 提取样本并重塑
                test_samples = x_test[:num_samples].reshape(min(num_samples, x_test.shape[0]), -1)
            else:
                # 尝试使用tensorflow-datasets加载
                try:
                    import tensorflow_datasets as tfds
                    dataset = tfds.load('celeb_a', split='test')
                    x_test = []
                    for example in dataset.take(num_samples):
                        # 调整为标准大小并规范化
                        img = example['image'].numpy()
                        # 如需调整大小（例如，到64x64）
                        from skimage.transform import resize
                        img = resize(img, (64, 64, 3), anti_aliasing=True, preserve_range=True)
                        img = img.astype('float32') / 255.0
                        x_test.append(img)
                    x_test = np.array(x_test)
                    
                    # 重塑为向量形式
                    x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
                    
                    # 提取样本
                    test_samples = x_test_reshaped[:num_samples]
                except:
                    raise ValueError(f"Failed to load CelebA data. Please download it to {celeba_dir} first.")
                    
        # 根据需要添加对其他数据集的支持
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # 保存为pickle文件
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

# 保留旧函数以向后兼容
def extract_mnist_test_data(num_samples=10, force_update=False):
    """向后兼容的遗留函数"""
    return extract_test_data("mnist", num_samples, force_update)

def load_test_data(dataset_name="mnist"):
    """
    加载已保存的测试数据
    
    Args:
        dataset_name: 数据集名称
    
    Returns:
        np.ndarray: 测试数据样本，如果未找到则为None
    """
    data_file = os.path.join("./data", f"{dataset_name}_test_data.pkl")
    if not os.path.exists(data_file):
        print(f"Warning: Test data file {data_file} not found")
        return None
        
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            
        # 检查数据是否为空
        if data is None or (isinstance(data, np.ndarray) and data.size == 0):
            print("Error: Test data is empty")
            return None
            
        # 确保数据是numpy数组
        if not isinstance(data, np.ndarray):
            print(f"Warning: Test data is not a numpy array, converting. Type: {type(data)}")
            data = np.array(data)
            
        # 根据数据集类型处理数据形状
        if data.ndim == 2:  # 已经是向量化形式 [batch_size, features]
            # 对于mnist、omniglot、binarymnist等
            if dataset_name.lower() in ["mnist", "omniglot", "binary_mnist", "binarymnist"]:
                img_dim = int(np.sqrt(data.shape[1]))
                if img_dim * img_dim == data.shape[1]:  # 确认是方形图像
                    # 保留原始格式，但也添加重塑后的版本以兼容
                    reshaped_data = data.reshape(data.shape[0], img_dim, img_dim)
                    return (data, reshaped_data)
            # 对于cifar10、svhn等
            elif dataset_name.lower() in ["cifar10", "svhn"]:
                if data.shape[1] == 32*32*3:  # 确认是32x32x3的图像
                    reshaped_data = data.reshape(data.shape[0], 32, 32, 3)
                    return (data, reshaped_data)
            # 对于celebA
            elif dataset_name.lower() in ["celeba"]:
                if data.shape[1] == 64*64*3:  # 确认是64x64x3的图像
                    reshaped_data = data.reshape(data.shape[0], 64, 64, 3)
                    return (data, reshaped_data)
                
        # 如果数据已经是多维的，返回原始数据和展平版本
        elif data.ndim >= 3:  # [batch_size, height, width, (channels)]
            flat_shape = data.shape[0], np.prod(data.shape[1:])
            flat_data = data.reshape(flat_shape)
            return (flat_data, data)
            
        # 如果以上都不适用，直接返回数据
        return data
        
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return None
