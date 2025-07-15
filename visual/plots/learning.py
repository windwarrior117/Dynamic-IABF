#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学习曲线可视化模块 - 提供学习曲线绘图功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..style_utils import COLORS, MARKERS, LINE_STYLES

class LearningVisualizer:
    """学习曲线可视化器，用于绘制和比较训练和验证损失"""
    
    def __init__(self, model_visualizer):
        """
        初始化学习曲线可视化器
        
        Args:
            model_visualizer: ModelVisualizer实例，提供数据访问方法
        """
        self.model_visualizer = model_visualizer
        self.base_dir = model_visualizer.base_dir
        self.model_dirs = model_visualizer.model_dirs
        self.model_names = model_visualizer.model_names
    
    def plot(self, save_path=None):
        """
        绘制学习曲线（训练和验证损失）
        
        Args:
            save_path: 保存图表的路径，如果为None则显示图表
        """
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        
        for i, model_dir in enumerate(self.model_dirs):
            epochs, train_losses, valid_losses, _ = self.model_visualizer._load_log(model_dir)
            if epochs and train_losses and valid_losses:
                # 使用一致风格绘制每个模型
                ax.plot(epochs, train_losses, marker=MARKERS[i], linestyle=LINE_STYLES[i], 
                       color=COLORS[i], label=f"{self.model_names[i]} - Training", 
                       markersize=6, markeredgewidth=1)
                
                ax.plot(epochs, valid_losses, marker=MARKERS[i], linestyle='--',
                       color=COLORS[i], alpha=0.7, label=f"{self.model_names[i]} - Validation", 
                       markerfacecolor='none', markeredgewidth=1, markevery=2)
        
        # 设置带有适当字体的标签
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        
        # 对y轴使用科学计数法（如果合适）
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
        
        # 更好的刻度间距
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 网格可读性
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 在图表周围添加边框（出版标准）
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        # 优化图例外观
        legend = ax.legend(frameon=True, framealpha=1, fancybox=False, edgecolor='black', loc='upper right')
        
        # 放置在出版物正确位置的标题
        ax.set_title('Training and Validation Loss Comparison')
        
        plt.tight_layout()
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 保存为高分辨率位图和矢量图像
            plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
            
            # 同样保存为出版用的矢量图形格式
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
                print(f"Learning curves saved to {save_path} and {vector_path}")
            else:
                print(f"Learning curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        return fig
