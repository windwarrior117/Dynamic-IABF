#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心模块 - 主可视化类和组织功能
"""

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

from .style_utils import set_publication_style, DATASET_PARAMS
from .data_utils import extract_test_data, load_test_data
from .model_utils import find_model_dirs, load_config

class ModelVisualizer:
    """模块化绘图工具，用于可视化和比较IABF、NECST和UAE模型性能"""
    
    def __init__(self, base_dir="./results/", model_dirs=None, model_names=None, dataset_name="mnist"):
        """
        初始化可视化工具
        
        Args:
            base_dir: 结果文件的基础目录
            model_dirs: 模型结果子目录列表，如 ["IABF/exp1", "NECST/exp1", "UAE/exp1"]
            model_names: 图例中使用的模型名称，如 ["IABF", "NECST", "UAE"]
            dataset_name: 被可视化的数据集名称 ('mnist', 'cifar10', 'svhn', 'omniglot', 'celebA')
        """
        self.base_dir = base_dir
        self.model_dirs = model_dirs if model_dirs else []
        self.model_names = model_names if model_names else ["Model " + str(i+1) for i in range(len(self.model_dirs))]
        self.dataset_name = dataset_name.lower()
        
        # 确保模型名称和目录数量匹配
        assert len(self.model_dirs) == len(self.model_names), "Model directory and name counts don't match"
        
        # 设置出版质量的绘图样式
        set_publication_style()
        
        # 尝试提取测试数据（如果尚未提取）
        self._ensure_test_data_available()
    
    def _ensure_test_data_available(self):
        """确保当前数据集的测试数据文件可用"""
        extract_test_data(self.dataset_name)
        
    def _load_log(self, model_dir):
        """提取训练和验证损失及互信息 (MI) 从日志文件"""
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
                    # 提取Epoch编号
                    epoch = int(parts[0].split()[1])
                    epochs.append(epoch)
                    
                    # 提取训练损失
                    for part in parts:
                        if "train loss" in part:
                            train_loss = float(part.split(':')[1].strip())
                            train_losses.append(train_loss)
                        elif "valid loss" in part and "time" not in part:
                            valid_loss = float(part.split(':')[1].strip())
                            valid_losses.append(valid_loss)
                        elif "mi" in part:
                            try:
                                mi = float(part.split(':')[1].strip())
                                mi_values.append(mi)
                            except ValueError:
                                print(f"Warning: Failed to parse MI value in {part}")
                
                # 从测试中提取互信息值
                if "mutual information" in line:
                    try:
                        parts = line.split()
                        mi_index = parts.index("information") + 1
                        if mi_index < len(parts):
                            mi_values.append(float(parts[mi_index]))
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Failed to parse mutual information from line: {line}")
        
        # 确保数组长度一致
        min_length = min(len(epochs), len(train_losses), len(valid_losses))
        if min_length < len(epochs):
            epochs = epochs[:min_length]
        if min_length < len(train_losses):
            train_losses = train_losses[:min_length]
        if min_length < len(valid_losses):
            valid_losses = valid_losses[:min_length]
        
        # 对于互信息，可能来自训练过程记录或测试结果，尝试保持与epoch数量一致
        if len(mi_values) > len(epochs) and len(epochs) > 0:
            # 通常测试MI添加在最后，我们截断到与训练数据相同的长度
            mi_values = mi_values[:len(epochs)]
        elif len(mi_values) < len(epochs) and len(mi_values) > 0:
            # 如果MI值少于epochs，我们用最后一个值填充（或者可以用None）
            last_mi = mi_values[-1]
            mi_values.extend([last_mi] * (len(epochs) - len(mi_values)))
            
        return epochs, train_losses, valid_losses, mi_values
    
    def _load_pickle(self, model_dir, filename="reconstr.pkl"):
        """加载重建数据"""
        filepath = os.path.join(self.base_dir, model_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: Pickle file {filepath} does not exist")
            return None
            
        with open(filepath, 'rb') as f:
            return pickle.load(f)
            
    # 以下是委托到专门可视化类的方法
    # 实现各可视化模块后将更新这些方法
    
    def plot_learning_curves(self, save_path=None):
        """绘制学习曲线（训练和验证损失）"""
        from .plots.learning import LearningVisualizer
        visualizer = LearningVisualizer(self)
        return visualizer.plot(save_path)
    
    def plot_mutual_information(self, save_path=None):
        """绘制互信息比较"""
        from .plots.mutual_information import MutualInformationVisualizer
        visualizer = MutualInformationVisualizer(self)
        return visualizer.plot(save_path)
    
    def plot_mutual_information_over_time(self, save_path=None, log_scale=False, smoothing=0.0):
        """绘制互信息随训练轮次的变化趋势"""
        from .plots.mutual_information import MutualInformationOverTimeVisualizer
        visualizer = MutualInformationOverTimeVisualizer(self)
        return visualizer.plot(save_path, log_scale, smoothing)
    
    def plot_distribution_comparison(self, save_path=None):
        """比较模型的边缘分布"""
        from .plots.distribution import DistributionVisualizer
        visualizer = DistributionVisualizer(self)
        return visualizer.plot(save_path)
    
    def plot_reconstruction_samples(self, num_samples=5, save_path=None):
        """使用PSNR指标比较模型间的重建样本"""
        from .plots.reconstruction import ReconstructionVisualizer
        visualizer = ReconstructionVisualizer(self)
        return visualizer.plot(num_samples, save_path)
    
    def plot_test_metrics(self, save_path=None):
        """绘制测试指标比较（测试损失、互信息等）"""
        from .plots.metrics import MetricsVisualizer
        visualizer = MetricsVisualizer(self)
        return visualizer.plot(save_path)
    
    def plot_noise_analysis(self, save_path=None):
        """分析不同噪声水平下的模型性能"""
        from .plots.metrics import NoiseAnalysisVisualizer
        visualizer = NoiseAnalysisVisualizer(self)
        return visualizer.plot(save_path)
    
    def export_test_metrics(self, csv_path=None):
        """
        导出所有模型的测试指标到CSV文件
        
        Args:
            csv_path: CSV文件保存路径，如果为None则仅返回DataFrame
            
        Returns:
            pandas.DataFrame: 包含测试指标的DataFrame
        """
        from .plots.metrics import MetricsVisualizer
        visualizer = MetricsVisualizer(self)
        return visualizer.export_metrics_table(csv_path)
    
    def plot_dynamic_iabf_features(self, save_path=None):
        """DynamicIABF模型的特性可视化"""
        from .plots.dynamic_iabf import DynamicIABFVisualizer
        visualizer = DynamicIABFVisualizer(self)
        return visualizer.plot(save_path)
    
    def collect_reconstruction_errors(self, csv_save_path=None):
        """
        从所有模型收集重建误差指标并保存到CSV
        
        Args:
            csv_save_path: 保存CSV文件的路径
            
        Returns:
            pandas.DataFrame: 带有重建误差指标的DataFrame
        """
        from .plots.reconstruction import ErrorTableVisualizer
        visualizer = ErrorTableVisualizer(self)
        return visualizer.collect_errors(csv_save_path)
    
    def plot_reconstruction_errors_table(self, save_path=None, csv_path=None):
        """
        创建重建误差指标的可视化表格
        
        Args:
            save_path: 保存表格图像的路径
            csv_path: 保存或加载CSV数据的路径（如果为None，数据将被收集但不保存到CSV）
            
        Returns:
            matplotlib.figure.Figure: 带有表格的Figure对象
        """
        from .plots.reconstruction import ErrorTableVisualizer
        visualizer = ErrorTableVisualizer(self)
        return visualizer.plot(save_path, csv_path)
    
    def plot_all(self, output_dir="./plots/"):
        """生成所有支持的图表"""
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Generating plots in {output_dir}...")
        
        # 按需调用各种绘图方法
        self.plot_learning_curves(os.path.join(output_dir, 'learning_curves.png'))
        self.plot_mutual_information(os.path.join(output_dir, 'mutual_information.png'))
        self.plot_mutual_information_over_time(os.path.join(output_dir, 'mutual_information_over_time.png'))
        self.plot_distribution_comparison(os.path.join(output_dir, 'distribution_comparison.png'))
        self.plot_reconstruction_samples(save_path=os.path.join(output_dir, 'reconstruction_samples.png'))
        self.plot_test_metrics(os.path.join(output_dir, 'test_metrics.png'))
        self.plot_noise_analysis(os.path.join(output_dir, 'noise_analysis.png'))
        self.plot_reconstruction_errors_table(
            save_path=os.path.join(output_dir, 'reconstruction_errors_table.png')
        )
        
        # 检查是否有DynamicIABF模型，如果有则绘制特性分析
        has_dynamic_model = False
        for model_dir in self.model_dirs:
            config = load_config(model_dir, self.base_dir)
            from .model_utils import identify_model_type
            if config and identify_model_type(config) == "DynamicIABF":
                has_dynamic_model = True
                break
        
        if has_dynamic_model:
            self.plot_dynamic_iabf_features(os.path.join(output_dir, 'dynamic_iabf_features.png'))
        
        print(f"All plots generated successfully in {output_dir}")
