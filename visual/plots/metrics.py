#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试指标可视化模块 - 提供模型测试指标比较和噪声分析功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

from ..style_utils import COLORS, MARKERS, LINE_STYLES

class MetricsVisualizer:
    """测试指标可视化器，用于比较不同模型的测试性能指标"""
    
    def __init__(self, model_visualizer):
        """
        初始化测试指标可视化器
        
        Args:
            model_visualizer: ModelVisualizer实例，提供数据访问方法
        """
        self.model_visualizer = model_visualizer
        self.base_dir = model_visualizer.base_dir
        self.model_dirs = model_visualizer.model_dirs
        self.model_names = model_visualizer.model_names
        self.dataset_name = model_visualizer.dataset_name
    
    def _load_test_metrics(self, model_dir):
        """
        从模型目录加载测试指标数据
        
        Args:
            model_dir: 模型结果目录
            
        Returns:
            dict: 包含测试指标的字典，如果数据不可用则返回None
        """
        # 首先尝试加载metrics.pkl
        metrics = self.model_visualizer._load_pickle(model_dir, "metrics.pkl")
        if metrics is not None and isinstance(metrics, dict):
            return metrics
            
        # 尝试从log.txt加载测试指标
        filepath = os.path.join(self.base_dir, model_dir, "log.txt")
        if not os.path.exists(filepath):
            return None
            
        # 解析日志文件中的测试指标
        test_metrics = {}
        with open(filepath, 'r') as f:
            for line in f:
                # 查找包含"Test"或"test"字样的行
                if "Test " in line or "test " in line:
                    parts = line.split(',')
                    for part in parts:
                        if ':' in part:
                            key, value = part.split(':', 1)
                            key = key.strip().lower()
                            value = value.strip()
                            try:
                                value = float(value)
                                test_metrics[key] = value
                            except ValueError:
                                # 如果不能转换为float，则保存为字符串
                                test_metrics[key] = value
                                
                # 特殊处理互信息值
                if "mutual information" in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "information" in part.lower() and i+1 < len(parts):
                            try:
                                test_metrics["mi"] = float(parts[i+1])
                            except ValueError:
                                pass
        
        return test_metrics if test_metrics else None
    
    def plot(self, save_path=None, metrics_to_show=None):
        """
        绘制模型测试指标比较图
        
        Args:
            save_path: 保存图表的路径，如果为None则显示图表
            metrics_to_show: 要显示的指标列表，如["loss", "accuracy", "mi"]，
                           如果为None则显示所有可用指标
            
        Returns:
            matplotlib.figure.Figure: 生成的图表对象
        """
        # 收集所有模型的测试指标
        all_metrics = []
        for model_dir in self.model_dirs:
            metrics = self._load_test_metrics(model_dir)
            all_metrics.append(metrics)
        
        # 如果没有任何测试指标，则返回提示
        if not any(all_metrics) or all(m is None for m in all_metrics):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No Test Metrics Available", 
                    fontsize=14, ha='center', va='center')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Empty test metrics chart saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            return fig
        
        # 确定所有可用指标
        available_metrics = set()
        for metrics in all_metrics:
            if metrics:
                available_metrics.update(metrics.keys())
        
        # 过滤出支持的指标类型，去除非数值型指标
        numeric_metrics = []
        for metric_name in available_metrics:
            for metrics in all_metrics:
                if metrics and metric_name in metrics and isinstance(metrics[metric_name], (int, float)):
                    numeric_metrics.append(metric_name)
                    break
        
        # 如果指定了metrics_to_show，则仅保留这些指标
        if metrics_to_show:
            numeric_metrics = [m for m in numeric_metrics if m in metrics_to_show]
        
        # 对指标进行分类和排序
        priority_metrics = ["loss", "val_loss", "test_loss", "accuracy", "val_accuracy", "test_accuracy", "mi"]
        numeric_metrics = sorted(numeric_metrics, key=lambda x: 
                               (priority_metrics.index(x) if x in priority_metrics else float('inf'), x))
        
        # 决定图表布局
        n_metrics = len(numeric_metrics)
        if n_metrics == 0:
            # 没有数值型指标可以显示
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No Numeric Test Metrics Available", 
                    fontsize=14, ha='center', va='center')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Empty test metrics chart saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            return fig
        
        # 确定图表布局
        if n_metrics == 1:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            axes = [axes]
        elif n_metrics == 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        else:
            # 计算最接近的长宽比为2:3的网格
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols/2, 5*n_rows/2))
            axes = axes.flatten()
        
        # 确保axes是列表
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        
        # 创建表格数据，用于最后输出到控制台
        table_data = [["Model"] + [metric.upper() for metric in numeric_metrics]]
        
        # 为每个指标绘制对比图
        for i, metric_name in enumerate(numeric_metrics):
            ax = axes[i]
            
            # 设置指标名称的标准化和显示
            display_name = metric_name.replace('_', ' ').title()
            if metric_name.lower() in ["mi", "mutual_information", "mutual information"]:
                display_name = "Mutual Information"
            elif metric_name.lower() in ["mse", "mean_squared_error"]:
                display_name = "Mean Squared Error"
            elif metric_name.lower() in ["mae", "mean_absolute_error"]:
                display_name = "Mean Absolute Error"
            
            # 为每个模型绘制指标条形图
            values = []
            for j, (metrics, model_name) in enumerate(zip(all_metrics, self.model_names)):
                if metrics and metric_name in metrics and isinstance(metrics[metric_name], (int, float)):
                    value = metrics[metric_name]
                    values.append(value)
                    ax.bar(j, value, color=COLORS[j % len(COLORS)], 
                          label=model_name, alpha=0.7, edgecolor='black', linewidth=1)
                    
                    # 在条形顶部显示数值
                    ax.text(j, value, f"{value:.4f}", ha='center', va='bottom', fontsize=9)
                else:
                    values.append(float('nan'))
                    ax.bar(j, 0, color=COLORS[j % len(COLORS)], 
                          label=model_name, alpha=0.1, edgecolor='black', linewidth=1, hatch='//')
                    ax.text(j, 0.1, "N/A", ha='center', va='bottom', fontsize=9, 
                          color='red', rotation=45)
            
            # 将数据添加到表格
            if i == 0:  # 只在处理第一个指标时添加模型名称行
                for j, model_name in enumerate(self.model_names):
                    if j < len(values):
                        value_str = f"{values[j]:.4f}" if not np.isnan(values[j]) else "N/A"
                        table_data.append([model_name, value_str])
                    else:
                        table_data.append([model_name, "N/A"])
            else:  # 对于后续指标，只添加值到已有行
                for j, _ in enumerate(self.model_names):
                    if j < len(values) and j+1 < len(table_data):
                        value_str = f"{values[j]:.4f}" if not np.isnan(values[j]) else "N/A"
                        table_data[j+1].append(value_str)
                    elif j+1 < len(table_data):
                        table_data[j+1].append("N/A")
            
            # 设置坐标轴和标签
            ax.set_title(display_name, fontsize=12)
            ax.set_xticks(range(len(self.model_names)))
            ax.set_xticklabels(self.model_names, rotation=45, ha='right')
            
            # 根据指标类型添加适当的Y轴标签
            if "loss" in metric_name.lower():
                ax.set_ylabel("Loss Value")
            elif "accuracy" in metric_name.lower() or "acc" == metric_name.lower():
                ax.set_ylabel("Accuracy")
                ax.set_ylim(0, 1.05)  # 通常准确率在0-1之间
            elif "error" in metric_name.lower():
                ax.set_ylabel("Error Value")
            elif metric_name.lower() in ["mi", "mutual_information", "mutual information"]:
                ax.set_ylabel("Bits")
            else:
                ax.set_ylabel("Value")
            
            # 添加网格线以便于阅读
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # 确保所有图表的y轴从0开始（除非全为负值）
            if not all(v < 0 for v in values if not np.isnan(v)):
                ylim = ax.get_ylim()
                ax.set_ylim(bottom=0, top=ylim[1])
                
            # 美化图表
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # 如果有空闲的子图，隐藏它们
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
        
        # 添加总标题
        plt.suptitle(f"Test Metrics Comparison - {self.dataset_name.upper()}", 
                    fontsize=14, y=0.98)
        
        # 使用fig.legend()在图表底部添加一个统一的图例
        handles = [Patch(facecolor=COLORS[i % len(COLORS)], 
                        edgecolor='black', label=self.model_names[i])
                 for i in range(len(self.model_names))]
        fig.legend(handles=handles, loc='lower center', ncol=min(5, len(self.model_names)), 
                 bbox_to_anchor=(0.5, 0.01), frameon=True, fancybox=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # 为图例留出空间
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 保存高分辨率图像
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # 也保存矢量格式
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight')
                print(f"Test metrics comparison saved to {save_path} and {vector_path}")
            else:
                print(f"Test metrics comparison saved to {save_path}")
            
            # 输出表格到控制台，便于快速查看
            print("\nTest Metrics Summary:")
            if table_data and len(table_data) > 0 and len(table_data[0]) > 0:
                col_widths = [max(len(row[i]) for row in table_data) for i in range(len(table_data[0]))]
                for i, row in enumerate(table_data):
                    print(" | ".join(val.ljust(col_widths[j]) for j, val in enumerate(row)))
                    if i == 0:  # 表头后添加分隔线
                        print("-" * sum(col_widths + [3] * (len(col_widths) - 1)))
            else:
                print("No metrics data available to display in table format.")
        else:
            plt.show()
        
        plt.close()
        return fig
    
    def export_metrics_table(self, csv_path=None):
        """
        将测试指标导出为数据表格
        
        Args:
            csv_path: CSV文件保存路径，如果为None则仅返回DataFrame
            
        Returns:
            pandas.DataFrame: 包含测试指标的DataFrame
        """
        # 收集所有模型的测试指标
        metrics_data = []
        
        for i, model_dir in enumerate(self.model_dirs):
            metrics = self._load_test_metrics(model_dir)
            model_data = {"Model": self.model_names[i]}
            
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        model_data[key] = value
            
            metrics_data.append(model_data)
        
        # 创建DataFrame
        df = pd.DataFrame(metrics_data)
        
        # 如果提供了CSV路径，保存到文件
        if csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"Test metrics exported to {csv_path}")
        
        return df


class NoiseAnalysisVisualizer:
    """噪声分析可视化器，用于分析不同噪声水平下的模型性能"""
    
    def __init__(self, model_visualizer):
        """
        初始化噪声分析可视化器
        
        Args:
            model_visualizer: ModelVisualizer实例，提供数据访问方法
        """
        self.model_visualizer = model_visualizer
        self.base_dir = model_visualizer.base_dir
        self.model_dirs = model_visualizer.model_dirs
        self.model_names = model_visualizer.model_names
        self.dataset_name = model_visualizer.dataset_name
    
    def _load_noise_analysis(self, model_dir):
        """
        从模型目录加载噪声分析数据
        
        Args:
            model_dir: 模型结果目录
            
        Returns:
            dict: 包含噪声分析的字典，如果数据不可用则返回None
        """
        # 首先尝试加载noise_analysis.pkl
        noise_data = self.model_visualizer._load_pickle(model_dir, "noise_analysis.pkl")
        
        # 如果文件存在但结构不正确，进行标准化
        if noise_data is not None:
            # 标准化噪声分析数据的格式
            if isinstance(noise_data, dict):
                # 已经是字典格式，检查结构
                if "noise_levels" in noise_data and "metrics" in noise_data:
                    return noise_data
                
                # 尝试重构标准格式
                standard_data = {"noise_levels": [], "metrics": {}}
                for key, value in noise_data.items():
                    try:
                        # 检查key是否可以解析为噪声水平
                        noise_level = float(key)
                        standard_data["noise_levels"].append(noise_level)
                        if isinstance(value, dict):
                            for metric_name, metric_value in value.items():
                                if metric_name not in standard_data["metrics"]:
                                    standard_data["metrics"][metric_name] = []
                                standard_data["metrics"][metric_name].append(metric_value)
                    except ValueError:
                        # 不是噪声水平的键，可能是指标名
                        if isinstance(value, list) and "noise_levels" not in standard_data:
                            standard_data["metrics"][key] = value
                            
                if standard_data["noise_levels"] and standard_data["metrics"]:
                    # 确保所有列表长度相同
                    list_length = len(standard_data["noise_levels"])
                    for metric_name, values in standard_data["metrics"].items():
                        if len(values) != list_length:
                            values.extend([float('nan')] * (list_length - len(values)))
                    return standard_data
            
            elif isinstance(noise_data, list) or isinstance(noise_data, tuple):
                # 列表或元组格式，尝试转换
                if len(noise_data) >= 2:
                    # 假设第一个元素是噪声水平，第二个是性能指标
                    noise_levels = noise_data[0]
                    if isinstance(noise_levels, (list, np.ndarray)):
                        standard_data = {"noise_levels": noise_levels, "metrics": {}}
                        
                        # 如果第二个元素是字典，直接使用
                        if isinstance(noise_data[1], dict):
                            standard_data["metrics"] = noise_data[1]
                        # 如果第二个元素是列表，假设是主要指标的值
                        elif isinstance(noise_data[1], (list, np.ndarray)) and len(noise_data[1]) == len(noise_levels):
                            standard_data["metrics"]["performance"] = noise_data[1]
                        
                        return standard_data
        
        return None
    
    def plot(self, save_path=None, metric_name=None):
        """
        绘制不同噪声水平下的模型性能分析图
        
        Args:
            save_path: 保存图表的路径，如果为None则显示图表
            metric_name: 要显示的指标名称，如果为None则尝试使用一个通用指标
            
        Returns:
            matplotlib.figure.Figure: 生成的图表对象
        """
        # 收集所有模型的噪声分析数据
        all_noise_data = []
        for model_dir in self.model_dirs:
            noise_data = self._load_noise_analysis(model_dir)
            all_noise_data.append(noise_data)
        
        # 检查是否有任何有效数据
        if not any(all_noise_data) or all(d is None for d in all_noise_data):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No Noise Analysis Data Available", 
                    fontsize=14, ha='center', va='center')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Empty noise analysis chart saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            return fig
        
        # 查找所有可用的指标名称
        available_metrics = set()
        for data in all_noise_data:
            if data and "metrics" in data:
                available_metrics.update(data["metrics"].keys())
        
        # 如果未指定指标名称，选择一个通用的指标
        if metric_name is None:
            # 优先选择误差相关指标
            for candidate in ["psnr", "ssim", "loss", "error", "mse", "accuracy", "performance"]:
                if candidate in available_metrics:
                    metric_name = candidate
                    break
            
            # 如果没找到，使用第一个可用指标
            if metric_name is None and available_metrics:
                metric_name = list(available_metrics)[0]
        
        # 如果仍然没有可用指标，显示错误信息
        if not metric_name or not available_metrics:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No Valid Metrics Found in Noise Analysis Data", 
                    fontsize=14, ha='center', va='center')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Empty noise analysis chart saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            return fig
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 设置指标标题
        metric_display_name = metric_name.replace('_', ' ').title()
        if metric_name.lower() in ["psnr"]:
            metric_display_name = "PSNR (dB)"
            y_label = "PSNR (dB)"
        elif metric_name.lower() in ["ssim"]:
            metric_display_name = "SSIM"
            y_label = "SSIM"
        elif "loss" in metric_name.lower():
            y_label = "Loss Value"
        elif "error" in metric_name.lower() or metric_name.lower() in ["mse", "mae"]:
            y_label = "Error Value"
        elif "accuracy" in metric_name.lower() or metric_name.lower() == "acc":
            y_label = "Accuracy"
        else:
            y_label = "Value"
        
        # 绘制每个模型的噪声分析曲线
        for i, (noise_data, model_name) in enumerate(zip(all_noise_data, self.model_names)):
            if noise_data and "noise_levels" in noise_data and "metrics" in noise_data:
                noise_levels = noise_data["noise_levels"]
                
                # 查找指定指标或回退到可用指标
                if metric_name in noise_data["metrics"]:
                    metric_values = noise_data["metrics"][metric_name]
                elif available_metrics and list(available_metrics)[0] in noise_data["metrics"]:
                    # 使用第一个可用指标
                    fallback_metric = list(available_metrics)[0]
                    metric_values = noise_data["metrics"][fallback_metric]
                    print(f"Warning: Metric '{metric_name}' not found for model '{model_name}'. Using '{fallback_metric}' instead.")
                else:
                    continue
                
                # 确保数据长度匹配
                if len(noise_levels) != len(metric_values):
                    min_len = min(len(noise_levels), len(metric_values))
                    noise_levels = noise_levels[:min_len]
                    metric_values = metric_values[:min_len]
                
                # 绘制线条
                ax.plot(noise_levels, metric_values, 
                       marker=MARKERS[i % len(MARKERS)],
                       linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                       color=COLORS[i % len(COLORS)],
                       label=model_name,
                       linewidth=2,
                       markersize=8)
        
        # 添加图表标题和标签
        ax.set_title(f"Noise Robustness Analysis - {metric_display_name}", fontsize=14)
        ax.set_xlabel("Noise Level", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 添加图例
        ax.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True)
        
        # 设置合理的y轴范围
        if "accuracy" in metric_name.lower() or metric_name.lower() in ["ssim"]:
            ax.set_ylim(0, 1.05)
        elif metric_name.lower() in ["psnr"]:
            # PSNR通常为正值
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 保存高分辨率图像
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # 也保存矢量格式
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight')
                print(f"Noise analysis chart saved to {save_path} and {vector_path}")
            else:
                print(f"Noise analysis chart saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        return fig
