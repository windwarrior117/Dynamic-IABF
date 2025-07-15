#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DynamicIABF特性可视化模块 - 提供动态信息瓶颈框架特性分析功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from ..style_utils import COLORS, MARKERS, LINE_STYLES

class DynamicIABFVisualizer:
    """DynamicIABF特性可视化器，用于分析动态信息瓶颈框架的特性"""
    
    def __init__(self, model_visualizer):
        """
        初始化DynamicIABF特性可视化器
        
        Args:
            model_visualizer: ModelVisualizer实例，提供数据访问方法
        """
        self.model_visualizer = model_visualizer
        self.base_dir = model_visualizer.base_dir
        self.model_dirs = model_visualizer.model_dirs
        self.model_names = model_visualizer.model_names
        self.dataset_name = model_visualizer.dataset_name
        
        # 存储找到的DynamicIABF模型索引
        self.dynamic_model_indices = self._find_dynamic_models()
    
    def _find_dynamic_models(self):
        """
        查找是DynamicIABF类型的模型索引
        
        Returns:
            list: DynamicIABF模型的索引列表
        """
        from ..model_utils import identify_model_type, load_config
        
        dynamic_indices = []
        
        for i, model_dir in enumerate(self.model_dirs):
            config = load_config(model_dir, self.base_dir)
            if config and identify_model_type(config) == "DynamicIABF":
                dynamic_indices.append(i)
                
        return dynamic_indices
    
    def _load_dynamic_features(self, model_idx):
        """
        从DynamicIABF模型目录加载特性数据
        
        Args:
            model_idx: 模型索引
            
        Returns:
            dict: 包含DynamicIABF特性的字典，如果数据不可用则返回None
        """
        if model_idx >= len(self.model_dirs):
            return None
            
        model_dir = self.model_dirs[model_idx]
        
        # 尝试加载dynamic_features.pkl
        features = self.model_visualizer._load_pickle(model_dir, "dynamic_features.pkl")
        if features is not None:
            return features
            
        # 尝试加载beta_adaptation.pkl（可能包含β值适应信息）
        beta_data = self.model_visualizer._load_pickle(model_dir, "beta_adaptation.pkl")
        if beta_data is not None:
            return {"beta_adaptation": beta_data}
            
        # 尝试加载noise_adaptation.pkl（可能包含噪声水平适应信息）
        noise_data = self.model_visualizer._load_pickle(model_dir, "noise_adaptation.pkl")
        if noise_data is not None:
            return {"noise_adaptation": noise_data}
            
        # 尝试从log.txt中提取动态特性信息
        log_path = os.path.join(self.base_dir, model_dir, "log.txt")
        if os.path.exists(log_path):
            # 初始化用于存储提取的特性
            dynamic_features = {
                "epochs": [],
                "beta_values": [],
                "noise_levels": [],
                "mi_values": []
            }
            
            with open(log_path, 'r') as f:
                for line in f:
                    if "Epoch" in line and ("beta:" in line.lower() or "noise:" in line.lower()):
                        parts = line.strip().split(',')
                        
                        # 提取epoch
                        try:
                            epoch = int(parts[0].split()[1])
                            dynamic_features["epochs"].append(epoch)
                        except (IndexError, ValueError):
                            continue
                        
                        # 提取beta值
                        beta_value = None
                        for part in parts:
                            if "beta:" in part.lower():
                                try:
                                    beta_value = float(part.split(':')[1].strip())
                                    break
                                except (IndexError, ValueError):
                                    pass
                        dynamic_features["beta_values"].append(beta_value)
                        
                        # 提取噪声水平
                        noise_level = None
                        for part in parts:
                            if "noise:" in part.lower():
                                try:
                                    noise_level = float(part.split(':')[1].strip())
                                    break
                                except (IndexError, ValueError):
                                    pass
                        dynamic_features["noise_levels"].append(noise_level)
                        
                        # 提取互信息值
                        mi_value = None
                        for part in parts:
                            if "mi:" in part.lower():
                                try:
                                    mi_value = float(part.split(':')[1].strip())
                                    break
                                except (IndexError, ValueError):
                                    pass
                        dynamic_features["mi_values"].append(mi_value)
            
            # 检查是否成功提取了任何动态特性
            if dynamic_features["epochs"]:
                return dynamic_features
                
        return None
    
    def plot_beta_adaptation(self, save_path=None):
        """
        绘制β值适应过程的分析图
        
        Args:
            save_path: 保存图表的路径，如果为None则显示图表
            
        Returns:
            matplotlib.figure.Figure: 生成的图表对象
        """
        # 收集所有DynamicIABF模型的β值适应数据
        beta_data = []
        model_indices = []
        
        for i in self.dynamic_model_indices:
            features = self._load_dynamic_features(i)
            if features:
                if "beta_adaptation" in features:
                    beta_data.append(features["beta_adaptation"])
                    model_indices.append(i)
                elif "beta_values" in features and features["beta_values"] and any(v is not None for v in features["beta_values"]):
                    # 构建适应数据
                    if "epochs" in features and len(features["epochs"]) == len(features["beta_values"]):
                        beta_data.append({
                            "epochs": features["epochs"],
                            "beta_values": features["beta_values"]
                        })
                        model_indices.append(i)
        
        if not beta_data:
            # 没有找到β值适应数据
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No Beta Adaptation Data Available", 
                    fontsize=14, ha='center', va='center')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Empty beta adaptation chart saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            return fig
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制每个模型的β值适应曲线
        for i, (data, model_idx) in enumerate(zip(beta_data, model_indices)):
            model_name = self.model_names[model_idx]
            
            epochs = None
            beta_values = None
            
            # 提取数据，处理不同的格式
            if isinstance(data, dict):
                if "epochs" in data and "beta_values" in data:
                    epochs = data["epochs"]
                    beta_values = data["beta_values"]
                elif "beta" in data:
                    # 可能包含β值的列表
                    beta_values = data["beta"]
                    epochs = list(range(len(beta_values)))
            elif isinstance(data, (list, np.ndarray)):
                # 直接的β值列表
                beta_values = data
                epochs = list(range(len(beta_values)))
            
            if epochs is not None and beta_values is not None and len(epochs) == len(beta_values):
                # 过滤掉None值
                valid_data = [(e, b) for e, b in zip(epochs, beta_values) if b is not None]
                if valid_data:
                    valid_epochs, valid_betas = zip(*valid_data)
                    
                    ax.plot(valid_epochs, valid_betas, 
                           marker=MARKERS[i % len(MARKERS)],
                           linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                           color=COLORS[i % len(COLORS)],
                           label=model_name,
                           linewidth=2,
                           markersize=6)
        
        # 添加图表标题和标签
        ax.set_title("Beta Value Adaptation Over Training", fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Beta Value", fontsize=12)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 添加图例
        ax.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True)
        
        # 设置合理的y轴范围（β值通常大于0）
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
                print(f"Beta adaptation chart saved to {save_path} and {vector_path}")
            else:
                print(f"Beta adaptation chart saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        return fig
    
    def plot_noise_adaptation(self, save_path=None):
        """
        绘制噪声水平适应过程的分析图
        
        Args:
            save_path: 保存图表的路径，如果为None则显示图表
            
        Returns:
            matplotlib.figure.Figure: 生成的图表对象
        """
        # 收集所有DynamicIABF模型的噪声适应数据
        noise_data = []
        model_indices = []
        
        for i in self.dynamic_model_indices:
            features = self._load_dynamic_features(i)
            if features:
                if "noise_adaptation" in features:
                    noise_data.append(features["noise_adaptation"])
                    model_indices.append(i)
                elif "noise_levels" in features and features["noise_levels"] and any(v is not None for v in features["noise_levels"]):
                    # 构建适应数据
                    if "epochs" in features and len(features["epochs"]) == len(features["noise_levels"]):
                        noise_data.append({
                            "epochs": features["epochs"],
                            "noise_levels": features["noise_levels"]
                        })
                        model_indices.append(i)
        
        if not noise_data:
            # 没有找到噪声适应数据
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No Noise Adaptation Data Available", 
                    fontsize=14, ha='center', va='center')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Empty noise adaptation chart saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            return fig
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制每个模型的噪声水平适应曲线
        for i, (data, model_idx) in enumerate(zip(noise_data, model_indices)):
            model_name = self.model_names[model_idx]
            
            epochs = None
            noise_levels = None
            
            # 提取数据，处理不同的格式
            if isinstance(data, dict):
                if "epochs" in data and "noise_levels" in data:
                    epochs = data["epochs"]
                    noise_levels = data["noise_levels"]
                elif "noise" in data:
                    # 可能包含噪声水平的列表
                    noise_levels = data["noise"]
                    epochs = list(range(len(noise_levels)))
            elif isinstance(data, (list, np.ndarray)):
                # 直接的噪声水平列表
                noise_levels = data
                epochs = list(range(len(noise_levels)))
            
            if epochs is not None and noise_levels is not None and len(epochs) == len(noise_levels):
                # 过滤掉None值
                valid_data = [(e, n) for e, n in zip(epochs, noise_levels) if n is not None]
                if valid_data:
                    valid_epochs, valid_noise = zip(*valid_data)
                    
                    ax.plot(valid_epochs, valid_noise, 
                           marker=MARKERS[i % len(MARKERS)],
                           linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                           color=COLORS[i % len(COLORS)],
                           label=model_name,
                           linewidth=2,
                           markersize=6)
        
        # 添加图表标题和标签
        ax.set_title("Noise Level Adaptation Over Training", fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Noise Level", fontsize=12)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 添加图例
        ax.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True)
        
        # 设置合理的y轴范围（噪声水平通常大于0）
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
                print(f"Noise adaptation chart saved to {save_path} and {vector_path}")
            else:
                print(f"Noise adaptation chart saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        return fig
    
    def plot_information_plane(self, save_path=None):
        """
        绘制信息平面图，展示互信息与噪声/β值的关系
        
        Args:
            save_path: 保存图表的路径，如果为None则显示图表
            
        Returns:
            matplotlib.figure.Figure: 生成的图表对象
        """
        # 收集所有DynamicIABF模型的信息平面数据
        info_plane_data = []
        model_indices = []
        
        for i in self.dynamic_model_indices:
            features = self._load_dynamic_features(i)
            if features:
                # 检查是否有足够的数据绘制信息平面
                has_mi = "mi_values" in features and features["mi_values"] and any(v is not None for v in features["mi_values"])
                has_noise = "noise_levels" in features and features["noise_levels"] and any(v is not None for v in features["noise_levels"])
                has_beta = "beta_values" in features and features["beta_values"] and any(v is not None for v in features["beta_values"])
                
                if has_mi and (has_noise or has_beta):
                    info_plane_data.append(features)
                    model_indices.append(i)
        
        if not info_plane_data:
            # 没有找到信息平面数据
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No Information Plane Data Available", 
                    fontsize=14, ha='center', va='center')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Empty information plane chart saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            return fig
        
        # 创建网格图表布局
        n_models = len(info_plane_data)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(6*n_cols, 5*n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # 为每个模型创建一个信息平面子图
        for idx, (data, model_idx) in enumerate(zip(info_plane_data, model_indices)):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            
            # 准备数据
            mi_values = data["mi_values"]
            has_noise = "noise_levels" in data and data["noise_levels"] and any(v is not None for v in data["noise_levels"])
            
            if has_noise:
                x_values = data["noise_levels"]
                x_label = "Noise Level"
            else:
                x_values = data["beta_values"]
                x_label = "Beta Value"
            
            # 确保数据长度一致
            min_len = min(len(mi_values), len(x_values))
            mi_values = mi_values[:min_len]
            x_values = x_values[:min_len]
            
            # 过滤掉None值
            valid_data = [(x, mi) for x, mi in zip(x_values, mi_values) if x is not None and mi is not None]
            
            if valid_data:
                valid_x, valid_mi = zip(*valid_data)
                
                # 构建点的颜色，表示训练轮次
                cmap = plt.cm.viridis
                colors = cmap(np.linspace(0, 1, len(valid_x)))
                
                # 使用散点图显示信息平面，颜色表示进度
                scatter = ax.scatter(valid_x, valid_mi, c=range(len(valid_x)), 
                                   cmap=cmap, s=50, alpha=0.8, edgecolors='k')
                
                # 添加连接点的线
                ax.plot(valid_x, valid_mi, '-', color='gray', alpha=0.5, zorder=0)
                
                # 标记起点和终点
                ax.scatter(valid_x[0], valid_mi[0], s=100, marker='o', 
                         facecolors='none', edgecolors='blue', linewidths=2, zorder=10)
                ax.scatter(valid_x[-1], valid_mi[-1], s=100, marker='*', 
                         color='red', edgecolors='k', zorder=10)
                
                # 添加颜色条，表示训练进度
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Training Progress')
                
                # 设置坐标轴标签和标题
                ax.set_xlabel(x_label, fontsize=12)
                ax.set_ylabel("Mutual Information (bits)", fontsize=12)
                ax.set_title(f"{self.model_names[model_idx]}", fontsize=14)
                
                # 添加网格
                ax.grid(True, linestyle='--', alpha=0.3)
                
                # 确保y轴从0开始
                ax.set_ylim(bottom=0)
                
                # 如果是噪声水平，确保x轴也从0开始
                if has_noise:
                    ax.set_xlim(left=0)
            else:
                ax.text(0.5, 0.5, "No Valid Data Points", 
                      fontsize=12, ha='center', va='center')
                ax.axis('off')
        
        # 如果有空白的子图，隐藏它们
        for idx in range(len(info_plane_data), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
        
        # 设置总标题
        fig.suptitle("Information Plane Analysis", fontsize=16, y=1.02)
        
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
                print(f"Information plane chart saved to {save_path} and {vector_path}")
            else:
                print(f"Information plane chart saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        return fig
    
    def plot_adaptation_summary(self, save_path=None):
        """
        绘制DynamicIABF模型的参数适应摘要，包括噪声、β值和互信息的变化
        
        Args:
            save_path: 保存图表的路径，如果为None则显示图表
            
        Returns:
            matplotlib.figure.Figure: 生成的图表对象
        """
        # 收集所有DynamicIABF模型的适应数据
        valid_data = []
        model_indices = []
        
        for i in self.dynamic_model_indices:
            features = self._load_dynamic_features(i)
            if features and "epochs" in features:
                # 检查是否有足够的数据
                valid_features = {"epochs": features["epochs"]}
                has_data = False
                
                if "beta_values" in features and features["beta_values"] and any(v is not None for v in features["beta_values"]):
                    valid_features["beta_values"] = features["beta_values"]
                    has_data = True
                    
                if "noise_levels" in features and features["noise_levels"] and any(v is not None for v in features["noise_levels"]):
                    valid_features["noise_levels"] = features["noise_levels"]
                    has_data = True
                    
                if "mi_values" in features and features["mi_values"] and any(v is not None for v in features["mi_values"]):
                    valid_features["mi_values"] = features["mi_values"]
                    has_data = True
                
                if has_data:
                    valid_data.append(valid_features)
                    model_indices.append(i)
        
        if not valid_data:
            # 没有找到有效数据
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No DynamicIABF Adaptation Data Available", 
                    fontsize=14, ha='center', va='center')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Empty adaptation summary chart saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            return fig
        
        # 创建网格图表布局
        n_models = len(valid_data)
        
        # 为每个模型创建一行，每行有1-3个子图（取决于有哪些数据）
        fig = plt.figure(figsize=(15, 5 * n_models))
        outer_grid = GridSpec(n_models, 1, figure=fig, hspace=0.3)
        
        for idx, (data, model_idx) in enumerate(zip(valid_data, model_indices)):
            model_name = self.model_names[model_idx]
            epochs = data["epochs"]
            
            # 计算这个模型有多少个指标
            n_metrics = 0
            has_beta = "beta_values" in data
            has_noise = "noise_levels" in data
            has_mi = "mi_values" in data
            
            if has_beta: n_metrics += 1
            if has_noise: n_metrics += 1
            if has_mi: n_metrics += 1
            
            # 创建子网格
            inner_grid = GridSpec(1, n_metrics, 
                                 wspace=0.3,
                                 subplot_spec=outer_grid[idx])
            
            # 当前子图的列索引
            col = 0
            
            # 为每个可用指标创建子图
            if has_beta:
                beta_ax = fig.add_subplot(inner_grid[0, col])
                col += 1
                
                beta_values = data["beta_values"]
                # 过滤掉None值
                valid_data = [(e, b) for e, b in zip(epochs, beta_values) if b is not None]
                
                if valid_data:
                    valid_epochs, valid_betas = zip(*valid_data)
                    
                    beta_ax.plot(valid_epochs, valid_betas, 
                               marker='o', linestyle='-',
                               color=COLORS[0 % len(COLORS)],
                               linewidth=2, markersize=6)
                    
                    beta_ax.set_title(f"Beta Value Adaptation", fontsize=12)
                    beta_ax.set_xlabel("Epoch", fontsize=10)
                    beta_ax.set_ylabel("Beta Value", fontsize=10)
                    beta_ax.grid(True, linestyle='--', alpha=0.3)
                    beta_ax.set_ylim(bottom=0)
                else:
                    beta_ax.text(0.5, 0.5, "No Valid Beta Data", 
                               fontsize=12, ha='center', va='center')
                    beta_ax.axis('off')
            
            if has_noise:
                noise_ax = fig.add_subplot(inner_grid[0, col])
                col += 1
                
                noise_levels = data["noise_levels"]
                # 过滤掉None值
                valid_data = [(e, n) for e, n in zip(epochs, noise_levels) if n is not None]
                
                if valid_data:
                    valid_epochs, valid_noise = zip(*valid_data)
                    
                    noise_ax.plot(valid_epochs, valid_noise, 
                                marker='s', linestyle='-',
                                color=COLORS[1 % len(COLORS)],
                                linewidth=2, markersize=6)
                    
                    noise_ax.set_title(f"Noise Level Adaptation", fontsize=12)
                    noise_ax.set_xlabel("Epoch", fontsize=10)
                    noise_ax.set_ylabel("Noise Level", fontsize=10)
                    noise_ax.grid(True, linestyle='--', alpha=0.3)
                    noise_ax.set_ylim(bottom=0)
                else:
                    noise_ax.text(0.5, 0.5, "No Valid Noise Data", 
                                fontsize=12, ha='center', va='center')
                    noise_ax.axis('off')
            
            if has_mi:
                mi_ax = fig.add_subplot(inner_grid[0, col])
                
                mi_values = data["mi_values"]
                # 过滤掉None值
                valid_data = [(e, mi) for e, mi in zip(epochs, mi_values) if mi is not None]
                
                if valid_data:
                    valid_epochs, valid_mi = zip(*valid_data)
                    
                    mi_ax.plot(valid_epochs, valid_mi, 
                             marker='^', linestyle='-',
                             color=COLORS[2 % len(COLORS)],
                             linewidth=2, markersize=6)
                    
                    mi_ax.set_title(f"Mutual Information", fontsize=12)
                    mi_ax.set_xlabel("Epoch", fontsize=10)
                    mi_ax.set_ylabel("MI (bits)", fontsize=10)
                    mi_ax.grid(True, linestyle='--', alpha=0.3)
                    mi_ax.set_ylim(bottom=0)
                else:
                    mi_ax.text(0.5, 0.5, "No Valid MI Data", 
                             fontsize=12, ha='center', va='center')
                    mi_ax.axis('off')
            
            # 添加模型名称作为行标题
            row_title_ax = fig.add_subplot(outer_grid[idx])
            row_title_ax.set_title(f"Model: {model_name}", fontsize=14, pad=30)
            row_title_ax.axis('off')
        
        # 设置总标题
        fig.suptitle("DynamicIABF Adaptation Analysis", fontsize=16, y=1.02)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 保存高分辨率图像
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # 也保存矢量格式
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight')
                print(f"Adaptation summary chart saved to {save_path} and {vector_path}")
            else:
                print(f"Adaptation summary chart saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        return fig
    
    def plot(self, save_path=None, plot_type="all"):
        """
        绘制DynamicIABF模型的特性分析图表
        
        Args:
            save_path: 保存图表的路径，如果为None则显示图表
            plot_type: 要绘制的图表类型，可选值：
                      "all" - 所有可用图表
                      "beta" - β值适应分析
                      "noise" - 噪声水平适应分析
                      "info_plane" - 信息平面分析
                      "summary" - 综合摘要
            
        Returns:
            matplotlib.figure.Figure: 生成的图表对象
        """
        # 检查是否有任何DynamicIABF模型
        if not self.dynamic_model_indices:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "No DynamicIABF Models Found", 
                    fontsize=14, ha='center', va='center')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Empty DynamicIABF features chart saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            return fig
        
        # 根据请求的图表类型调用相应的方法
        if plot_type == "beta":
            return self.plot_beta_adaptation(save_path)
        elif plot_type == "noise":
            return self.plot_noise_adaptation(save_path)
        elif plot_type == "info_plane":
            return self.plot_information_plane(save_path)
        elif plot_type == "summary":
            return self.plot_adaptation_summary(save_path)
        else:  # "all" or any other value
            # 创建包含所有图表的综合视图
            return self.plot_adaptation_summary(save_path)
