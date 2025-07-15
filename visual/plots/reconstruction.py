#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重建可视化模块 - 提供模型重建样本比较和误差指标可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from ..style_utils import COLORS, MARKERS, LINE_STYLES
from ..data_utils import load_test_data

class ReconstructionVisualizer:
    """重建样本可视化器，用于比较不同模型的样本重建效果"""
    
    def __init__(self, model_visualizer):
        """
        初始化重建样本可视化器
        
        Args:
            model_visualizer: ModelVisualizer实例，提供数据访问方法
        """
        self.model_visualizer = model_visualizer
        self.base_dir = model_visualizer.base_dir
        self.model_dirs = model_visualizer.model_dirs
        self.model_names = model_visualizer.model_names
        self.dataset_name = model_visualizer.dataset_name
    
    def _load_reconstructions(self, model_dir):
        """
        从模型目录加载重建样本数据
        
        Args:
            model_dir: 模型结果目录
            
        Returns:
            tuple: (原始图像, 重建图像) 或 None（如果数据不可用）
        """
        recon_data = self.model_visualizer._load_pickle(model_dir, "reconstr.pkl")
        if recon_data is None:
            return None, None
        
        # 检查是否为字典格式
        if isinstance(recon_data, dict):
            if 'original' in recon_data and 'reconstructed' in recon_data:
                originals = recon_data['original']
                reconstructed = recon_data['reconstructed']
                
                # 检查数据有效性
                if originals is None or reconstructed is None:
                    return None, None
                
                # 确保数据是numpy数组
                if not isinstance(originals, np.ndarray):
                    originals = np.array(originals)
                if not isinstance(reconstructed, np.ndarray):
                    reconstructed = np.array(reconstructed)
                    
                return originals, reconstructed
            else:
                # 尝试其他可能的键名组合
                possible_keys = [('x', 'x_recon'), ('x_test', 'x_recon'), ('originals', 'reconstructions')]
                for orig_key, recon_key in possible_keys:
                    if orig_key in recon_data and recon_key in recon_data:
                        originals = recon_data[orig_key]
                        reconstructed = recon_data[recon_key]
                        
                        # 检查数据有效性
                        if originals is None or reconstructed is None:
                            continue
                            
                        # 确保数据是numpy数组
                        if not isinstance(originals, np.ndarray):
                            originals = np.array(originals)
                        if not isinstance(reconstructed, np.ndarray):
                            reconstructed = np.array(reconstructed)
                            
                        return originals, reconstructed
        
        # 如果是元组格式，假设第一个元素是原始图像，第二个是重建图像
        elif isinstance(recon_data, tuple) and len(recon_data) >= 2:
            originals = recon_data[0]
            reconstructed = recon_data[1]
            
            # 检查数据有效性
            if originals is None or reconstructed is None:
                return None, None
                
            # 确保数据是numpy数组
            if not isinstance(originals, np.ndarray):
                originals = np.array(originals)
            if not isinstance(reconstructed, np.ndarray):
                reconstructed = np.array(reconstructed)
                
            return originals, reconstructed
        
        # 新增：如果直接是numpy数组，假设它是重建图像，尝试从测试数据加载原始图像
        elif isinstance(recon_data, np.ndarray) and recon_data.size > 0:
            # 加载测试数据作为原始图像
            test_data = load_test_data(self.dataset_name)
            if test_data is not None:
                # 处理test_data可能是元组的情况
                if isinstance(test_data, tuple) and len(test_data) >= 1:
                    originals = test_data[0]
                else:
                    originals = test_data
                
                # 确保数据形状匹配
                if len(originals) >= len(recon_data):
                    # 选择与重建数据相同数量的原始图像
                    originals = originals[:len(recon_data)]
                    return originals, recon_data
                else:
                    print(f"Warning: Not enough test samples for {self.dataset_name} to match reconstructions")
            
            # 如果没有找到测试数据，只返回重建结果
            print(f"Warning: No original test data found for {self.dataset_name}, returning reconstructions only")
            return None, recon_data
        
        return None, None
    
    def _compute_metrics(self, original, reconstructed):
        """
        计算重建图像的质量指标 (PSNR, SSIM)
        
        Args:
            original: 原始图像数组
            reconstructed: 重建图像数组
            
        Returns:
            tuple: (PSNR值, SSIM值)
        """
        if original is None or reconstructed is None or original.size == 0 or reconstructed.size == 0:
            return "N/A", "N/A"
            
        # 确保值域在[0, 1]之间，这是PSNR计算的要求
        if original.max() > 1.0:
            original = original / 255.0
        if reconstructed.max() > 1.0:
            reconstructed = reconstructed / 255.0
            
        # 计算平均PSNR
        psnr_vals = []
        ssim_vals = []
        
        for i in range(min(len(original), len(reconstructed))):
            orig_img = original[i]
            recon_img = reconstructed[i]
            
            # 处理多通道图像
            if len(orig_img.shape) == 3 and orig_img.shape[2] == 3:
                # 彩色图像 - 计算每个通道的PSNR并平均
                psnr_val = psnr(orig_img, recon_img, data_range=1.0)
                ssim_val = ssim(orig_img, recon_img, data_range=1.0, channel_axis=2)
            else:
                # 灰度图像
                psnr_val = psnr(orig_img, recon_img, data_range=1.0)
                ssim_val = ssim(orig_img, recon_img, data_range=1.0)
                
            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)
        
        avg_psnr = np.mean(psnr_vals) if psnr_vals else "N/A"
        avg_ssim = np.mean(ssim_vals) if ssim_vals else "N/A"
        
        return avg_psnr, avg_ssim
    
    def plot(self, num_samples=5, save_path=None):
        """
        比较不同模型的重建样本
        
        Args:
            num_samples: 要显示的样本数量
            save_path: 保存图表的路径，如果为None则显示图表
            
        Returns:
            matplotlib.figure.Figure: 生成的图表对象
        """
        # 确保至少有一个模型
        if not self.model_dirs:
            print("No model directories specified.")
            return None

        # 模型数量加上原始图像一行
        n_rows = len(self.model_dirs) + 1
        n_cols = num_samples
        
        # 创建适当大小的图表
        plt.figure(figsize=(2.5*n_cols, 2.2*n_rows))
        
        # 尝试加载每个模型的重建结果
        all_originals = []
        all_reconstructions = []
        has_data = False
        
        for model_dir in self.model_dirs:
            originals, reconstructions = self._load_reconstructions(model_dir)
            all_originals.append(originals)
            all_reconstructions.append(reconstructions)
            
            if originals is not None and reconstructions is not None:
                has_data = True
        
        # 如果没有任何模型有重建数据，尝试从测试数据加载样本
        if not has_data:
            try:
                test_data = load_test_data(self.dataset_name)
                if test_data is not None:
                    # 处理新的返回格式，可能是元组(flat_data, reshaped_data)
                    if isinstance(test_data, tuple) and len(test_data) >= 2:
                        flat_data, image_data = test_data
                        test_images = image_data[:num_samples]
                    else:
                        test_images = test_data[:num_samples]
                        
                    plt.suptitle("No Reconstruction Data Available - Showing Test Images Only", fontsize=16)
                    
                    for j in range(min(num_samples, len(test_images))):
                        plt.subplot(1, num_samples, j+1)
                        img = test_images[j]
                        
                        # 处理不同维度的图像
                        if img.ndim == 1:  # 向量形式，需要重塑
                            img_size = int(np.sqrt(img.size))
                            if img_size * img_size == img.size:  # 是方形图像
                                img = img.reshape(img_size, img_size)
                            elif img.size == 32*32*3:  # CIFAR/SVHN
                                img = img.reshape(32, 32, 3)
                            elif img.size == 64*64*3:  # CelebA
                                img = img.reshape(64, 64, 3)
                        elif img.ndim == 3:
                            if img.shape[0] == 1 or img.shape[0] == 3:  # Channel-first (1, H, W) or (3, H, W)
                                img = np.transpose(img, (1, 2, 0))
                            
                            if img.shape[2] == 1:  # (H, W, 1)
                                img = img.squeeze(-1)
                        
                        if img.max() > 1.0:
                            img = img / 255.0
                            
                        plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
                        plt.title(f"Sample {j+1}")
                        plt.axis('off')
                    
                    plt.tight_layout()
                    
                    if save_path:
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        print(f"Test images saved to {save_path}")
                    else:
                        plt.show()
                    
                    plt.close()
                    return plt.gcf()
            except Exception as e:
                print(f"Error loading test data: {str(e)}")
                plt.suptitle("No Reconstruction Data Available", fontsize=16)
                plt.axis('off')
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                else:
                    plt.show()
                    
                plt.close()
                return plt.gcf()
        
        # 确定可用的样本数量
        available_samples = float('inf')
        for originals, reconstructions in zip(all_originals, all_reconstructions):
            if reconstructions is not None:
                # 只要有重建图像就可以，不需要原始图像也存在
                available_samples = min(available_samples, len(reconstructions))
        
        if available_samples == float('inf'):
            available_samples = 0
            
        num_samples = min(num_samples, available_samples)
        
        if num_samples == 0:
            plt.suptitle("No Reconstruction Samples Available", fontsize=16)
            plt.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
            plt.close()
            return plt.gcf()
        
        # 设置图表
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5*n_cols, 2.2*n_rows))
        # 单行或单列时，确保axes是二维的
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        elif n_cols == 1:
            axes = np.expand_dims(axes, axis=1)
        
        # 创建表格存储PSNR和SSIM指标
        model_metrics = []
        
        # 绘制原始图像
        has_originals = False
        used_original_index = None
        
        # 尝试从所有模型中找到一个有原始图像的
        for i, originals in enumerate(all_originals):
            if originals is not None:
                has_originals = True
                used_original_index = i
                for j in range(num_samples):
                    if j < len(originals):
                        image = originals[j]
                        
                        # 处理不同维度的图像
                        if len(image.shape) == 1:  # 一维向量，需要重塑为2D图像
                            img_dim = int(np.sqrt(image.size))
                            if img_dim * img_dim == image.size:  # 确认是方形图像
                                image = image.reshape(img_dim, img_dim)
                            elif image.size == 32*32*3:  # CIFAR/SVHN
                                image = image.reshape(32, 32, 3)
                            elif image.size == 64*64*3:  # CelebA
                                image = image.reshape(64, 64, 3)
                        elif len(image.shape) == 3:
                            if image.shape[0] == 1 or image.shape[0] == 3:  # Channel-first (1, H, W) or (3, H, W)
                                image = np.transpose(image, (1, 2, 0))
                            
                            if image.shape[2] == 1:  # (H, W, 1)
                                image = image.squeeze(-1)
                        
                        if image.max() > 1.0:
                            image = image / 255.0
                            
                        axes[0, j].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
                        axes[0, j].set_title(f"Original {j+1}")
                        axes[0, j].axis('off')
                break
        
        # 如果没有原始图像，尝试使用测试数据
        if not has_originals:
            try:
                test_data = load_test_data(self.dataset_name)
                if test_data is not None:
                    # 处理新的返回格式，可能是元组(flat_data, reshaped_data)
                    if isinstance(test_data, tuple) and len(test_data) >= 2:
                        flat_data, image_data = test_data
                        test_images = image_data[:num_samples]
                    else:
                        test_images = test_data[:num_samples]
                        
                    for j in range(min(num_samples, len(test_images))):
                        img = test_images[j]
                        
                        # 处理不同维度的图像
                        if img.ndim == 1:  # 向量形式，需要重塑
                            img_size = int(np.sqrt(img.size))
                            if img_size * img_size == img.size:  # 是方形图像
                                img = img.reshape(img_size, img_size)
                            elif img.size == 32*32*3:  # CIFAR/SVHN
                                img = img.reshape(32, 32, 3)
                            elif img.size == 64*64*3:  # CelebA
                                img = img.reshape(64, 64, 3)
                        elif img.ndim == 3:
                            if img.shape[0] == 1 or img.shape[0] == 3:  # Channel-first
                                img = np.transpose(img, (1, 2, 0))
                            
                            if img.shape[2] == 1:  # (H, W, 1)
                                img = img.squeeze(-1)
                        
                        if img.max() > 1.0:
                            img = img / 255.0
                            
                        axes[0, j].imshow(img, cmap='gray' if img.ndim == 2 else None)
                        axes[0, j].set_title(f"Original {j+1}")
                        axes[0, j].axis('off')
            except Exception as e:
                print(f"Error loading original images: {str(e)}")
                # 第一行显示"无原始图像"
                for j in range(num_samples):
                    axes[0, j].text(0.5, 0.5, "No Original Image", 
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform=axes[0, j].transAxes,
                              fontsize=10)
                    axes[0, j].axis('off')
        
        # 为每个模型绘制重建图像
        for i in range(len(self.model_dirs)):
            if all_reconstructions[i] is None:
                for j in range(num_samples):
                    axes[i+1, j].text(0.5, 0.5, "No Data", 
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    transform=axes[i+1, j].transAxes,
                                    fontsize=12)
                    axes[i+1, j].axis('off')
                model_metrics.append({"Model": self.model_names[i], "PSNR": "N/A", "SSIM": "N/A"})
                continue
                
            model_psnr_vals = []
            model_ssim_vals = []
            
            for j in range(num_samples):
                orig_img = all_originals[i][j] if all_originals[i] is not None else None
                recon_img = all_reconstructions[i][j]
                
                # 处理不同维度的图像
                if len(recon_img.shape) == 1:  # 一维向量，需要重塑为2D图像
                    img_dim = int(np.sqrt(recon_img.size))
                    if img_dim * img_dim == recon_img.size:  # 确认是方形图像
                        recon_img = recon_img.reshape(img_dim, img_dim)
                    elif recon_img.size == 32*32*3:  # CIFAR/SVHN
                        recon_img = recon_img.reshape(32, 32, 3)
                    elif recon_img.size == 64*64*3:  # CelebA
                        recon_img = recon_img.reshape(64, 64, 3)
                elif len(recon_img.shape) == 3:
                    if recon_img.shape[0] == 1 or recon_img.shape[0] == 3:  # Channel-first
                        recon_img = np.transpose(recon_img, (1, 2, 0))
                    
                    if recon_img.shape[2] == 1:  # (H, W, 1)
                        recon_img = recon_img.squeeze(-1)
                
                if recon_img.max() > 1.0:
                    recon_img = recon_img / 255.0
                
                axes[i+1, j].imshow(recon_img, cmap='gray' if len(recon_img.shape) == 2 else None)
                
                # 计算PSNR和SSIM（如果有原始图像）
                if orig_img is not None:
                    # 获取与当前重建图像匹配的形状的原始图像
                    if len(orig_img.shape) == 1:  # 一维向量，需要重塑
                        img_dim = int(np.sqrt(orig_img.size))
                        if img_dim * img_dim == orig_img.size:  # 确认是方形图像
                            orig_img = orig_img.reshape(img_dim, img_dim)
                        elif orig_img.size == 32*32*3:  # CIFAR/SVHN
                            orig_img = orig_img.reshape(32, 32, 3)
                        elif orig_img.size == 64*64*3:  # CelebA
                            orig_img = orig_img.reshape(64, 64, 3)
                    elif len(orig_img.shape) == 3:
                        if orig_img.shape[0] == 1 or orig_img.shape[0] == 3:  # Channel-first
                            orig_img = np.transpose(orig_img, (1, 2, 0))
                        
                        if orig_img.shape[2] == 1:  # (H, W, 1)
                            orig_img = orig_img.squeeze(-1)
                    
                    if orig_img.max() > 1.0:
                        orig_img = orig_img / 255.0
                    
                    # 确保原始图像和重建图像具有相同的形状
                    if orig_img.shape == recon_img.shape:
                        if len(orig_img.shape) == 3 and orig_img.shape[2] == 3:  # 彩色图像
                            psnr_val = psnr(orig_img, recon_img, data_range=1.0)
                            ssim_val = ssim(orig_img, recon_img, data_range=1.0, channel_axis=2)
                        else:  # 灰度图像
                            psnr_val = psnr(orig_img, recon_img, data_range=1.0)
                            ssim_val = ssim(orig_img, recon_img, data_range=1.0)
                        
                        model_psnr_vals.append(psnr_val)
                        model_ssim_vals.append(ssim_val)
                        
                        # 在图像上显示PSNR值
                        psnr_text = f"PSNR: {psnr_val:.2f}dB"
                        ssim_text = f"SSIM: {ssim_val:.4f}"
                        axes[i+1, j].text(0.5, 0.05, psnr_text, 
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        transform=axes[i+1, j].transAxes, 
                                        color='white', fontsize=8,
                                        bbox=dict(facecolor='black', alpha=0.6))
                else:
                    # 如果没有原始图像，只显示重建图像，不计算指标
                    pass
                
                axes[i+1, j].set_title(f"{self.model_names[i]}")
                axes[i+1, j].axis('off')
            
            # 计算平均指标
            avg_psnr = np.mean(model_psnr_vals) if model_psnr_vals else "N/A"
            avg_ssim = np.mean(model_ssim_vals) if model_ssim_vals else "N/A"
            
            model_metrics.append({
                "Model": self.model_names[i],
                "PSNR": f"{avg_psnr:.2f}dB" if isinstance(avg_psnr, (int, float)) else avg_psnr,
                "SSIM": f"{avg_ssim:.4f}" if isinstance(avg_ssim, (int, float)) else avg_ssim
            })
        
        # 添加图表标题
        plt.suptitle(f"Reconstruction Quality Comparison - {self.dataset_name.upper()}", fontsize=16, y=1.02)
        
        # 在图表底部添加平均PSNR和SSIM指标表格
        if model_metrics:
            # 创建指标表格数据
            table_data = []
            for metric in model_metrics:
                if metric["PSNR"] != "N/A" or metric["SSIM"] != "N/A":
                    table_data.append([metric["Model"], metric["PSNR"], metric["SSIM"]])
            
            # 如果有有效指标，添加表格
            if table_data:
                # 获取表格轴位置
                table_ax = fig.add_axes([0.25, 0.02, 0.5, 0.1], frameon=False)
                table_ax.axis('off')
                
                # 创建表格
                table = table_ax.table(
                    cellText=table_data,
                    colLabels=["Model", "Avg. PSNR", "Avg. SSIM"],
                    loc='center',
                    cellLoc='center',
                    bbox=[0, 0, 1, 1]
                )
                
                # 设置表格样式
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                
                # 使表格标题加粗
                for (i, j), cell in table.get_celld().items():
                    if i == 0:  # 表头行
                        cell.set_text_props(weight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # 为表格留出空间
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 保存高分辨率图像
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # 也保存矢量格式
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight')
                print(f"Reconstruction samples saved to {save_path} and {vector_path}")
            else:
                print(f"Reconstruction samples saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        return fig


class ErrorTableVisualizer:
    """重建误差表格可视化器，用于比较模型间的重建误差指标"""
    
    def __init__(self, model_visualizer):
        """
        初始化重建误差表格可视化器
        
        Args:
            model_visualizer: ModelVisualizer实例，提供数据访问方法
        """
        self.model_visualizer = model_visualizer
        self.base_dir = model_visualizer.base_dir
        self.model_dirs = model_visualizer.model_dirs
        self.model_names = model_visualizer.model_names
        self.dataset_name = model_visualizer.dataset_name
    
    def collect_errors(self, csv_save_path=None):
        """
        收集所有模型的重建误差指标
        
        Args:
            csv_save_path: 保存CSV文件的路径，如果为None则不保存
            
        Returns:
            pandas.DataFrame: 包含重建误差指标的DataFrame
        """
        # 收集所有模型的误差指标
        recon_vis = ReconstructionVisualizer(self.model_visualizer)
        
        metrics = []
        
        for i, model_dir in enumerate(self.model_dirs):
            originals, reconstructions = recon_vis._load_reconstructions(model_dir)
            
            if originals is not None and reconstructions is not None:
                # 计算PSNR和SSIM
                psnr_val, ssim_val = recon_vis._compute_metrics(originals, reconstructions)
                
                # 加载额外的MSE和MAE指标（如果存在）
                reconstr_metrics = self.model_visualizer._load_pickle(model_dir, "metrics.pkl")
                
                mse = "N/A"
                mae = "N/A"
                
                if reconstr_metrics:
                    if isinstance(reconstr_metrics, dict):
                        mse = reconstr_metrics.get('mse', reconstr_metrics.get('MSE', "N/A"))
                        mae = reconstr_metrics.get('mae', reconstr_metrics.get('MAE', "N/A"))
                
                metrics.append({
                    'Model': self.model_names[i],
                    'PSNR': psnr_val if psnr_val != "N/A" else np.nan,
                    'SSIM': ssim_val if ssim_val != "N/A" else np.nan,
                    'MSE': mse if mse != "N/A" else np.nan,
                    'MAE': mae if mae != "N/A" else np.nan
                })
            else:
                metrics.append({
                    'Model': self.model_names[i],
                    'PSNR': np.nan,
                    'SSIM': np.nan,
                    'MSE': np.nan,
                    'MAE': np.nan
                })
        
        # 转换为DataFrame
        df = pd.DataFrame(metrics)
        
        # 保存到CSV（如果需要）
        if csv_save_path:
            os.makedirs(os.path.dirname(os.path.abspath(csv_save_path)), exist_ok=True)
            df.to_csv(csv_save_path, index=False)
            print(f"Reconstruction error metrics saved to {csv_save_path}")
        
        return df
    
    def plot(self, save_path=None, csv_path=None):
        """
        创建重建误差指标的可视化表格
        
        Args:
            save_path: 保存表格图像的路径，如果为None则显示图表
            csv_path: 加载或保存CSV数据的路径
            
        Returns:
            matplotlib.figure.Figure: 带有表格的Figure对象
        """
        # 如果提供了CSV路径并且文件存在，从文件加载数据
        if csv_path and os.path.exists(csv_path):
            try:
                metrics_df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error reading CSV file: {str(e)}")
                metrics_df = self.collect_errors(csv_path if csv_path != save_path else None)
        else:
            # 否则收集指标
            metrics_df = self.collect_errors(csv_path)
        
        if metrics_df.empty:
            print("No reconstruction error metrics available.")
            return None
        
        # 删除所有指标都为NaN的行
        metrics_df = metrics_df.dropna(how='all', subset=['PSNR', 'SSIM', 'MSE', 'MAE'])
        
        if metrics_df.empty:
            print("No valid reconstruction error metrics available after filtering.")
            return None
        
        # 确定要显示的列
        columns_to_display = ['Model', 'PSNR', 'SSIM']
        if not metrics_df['MSE'].isna().all():
            columns_to_display.append('MSE')
        if not metrics_df['MAE'].isna().all():
            columns_to_display.append('MAE')
        
        # 用于表格的数据
        table_data = []
        headers = columns_to_display.copy()
        
        for _, row in metrics_df.iterrows():
            row_data = []
            for col in columns_to_display:
                if col == 'Model':
                    row_data.append(row[col])
                elif pd.isna(row[col]):
                    row_data.append("N/A")
                elif col == 'PSNR':
                    row_data.append(f"{row[col]:.2f}dB")
                elif col == 'SSIM':
                    row_data.append(f"{row[col]:.4f}")
                else:  # MSE and MAE
                    row_data.append(f"{row[col]:.6f}")
            table_data.append(row_data)
        
        # 创建表格图表
        fig = plt.figure(figsize=(8, 3 + 0.6 * len(metrics_df)))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # 创建表格
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # 表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.8)
        
        # 设置单元格样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # 表头行
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E0E0E0')
            elif j == 0:  # 模型名称列
                cell.set_text_props(weight='bold')
            else:  # 指标列
                # 颜色映射 - 根据值的好坏设置背景色
                if i > 0 and j > 0 and table_data[i-1][j] != "N/A":
                    if headers[j] == 'PSNR' or headers[j] == 'SSIM':
                        # 这些指标越高越好
                        try:
                            val = float(table_data[i-1][j].replace('dB', ''))
                            # 找出此列所有数值中的最大值和最小值
                            all_vals = []
                            for row in table_data:
                                if row[j] != "N/A":
                                    all_vals.append(float(row[j].replace('dB', '')))
                            
                            if all_vals:
                                min_val = min(all_vals)
                                max_val = max(all_vals)
                                if max_val > min_val:
                                    # 归一化到[0, 1]
                                    norm_val = (val - min_val) / (max_val - min_val)
                                    # 从红到绿的渐变色
                                    rgb = plt.cm.RdYlGn(norm_val)
                                    cell.set_facecolor(rgb)
                        except:
                            pass
                    elif headers[j] == 'MSE' or headers[j] == 'MAE':
                        # 这些指标越低越好
                        try:
                            val = float(table_data[i-1][j])
                            # 找出此列所有数值中的最大值和最小值
                            all_vals = []
                            for row in table_data:
                                if row[j] != "N/A":
                                    all_vals.append(float(row[j]))
                            
                            if all_vals:
                                min_val = min(all_vals)
                                max_val = max(all_vals)
                                if max_val > min_val:
                                    # 归一化到[0, 1]，反转使低值变为好值
                                    norm_val = 1 - (val - min_val) / (max_val - min_val)
                                    # 从红到绿的渐变色
                                    rgb = plt.cm.RdYlGn(norm_val)
                                    cell.set_facecolor(rgb)
                        except:
                            pass
        
        # 添加标题
        plt.suptitle(f"Reconstruction Quality Metrics - {self.dataset_name.upper()}", 
                   fontsize=14, y=0.98)
        
        # 添加指标说明
        footnote = """
        PSNR: Peak Signal-to-Noise Ratio (higher is better)
        SSIM: Structural Similarity Index (higher is better)
        """
        if 'MSE' in headers:
            footnote += "MSE: Mean Squared Error (lower is better)\n"
        if 'MAE' in headers:
            footnote += "MAE: Mean Absolute Error (lower is better)"
            
        plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=10, 
                  bbox={"facecolor":"none", "edgecolor":"none", "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # 为脚注留出空间
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 保存高分辨率图像
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # 也保存矢量格式
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight')
                print(f"Reconstruction error table saved to {save_path} and {vector_path}")
            else:
                print(f"Reconstruction error table saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        return fig
