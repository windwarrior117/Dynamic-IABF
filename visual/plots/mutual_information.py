#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
互信息可视化模块 - 提供模型间互信息比较功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..style_utils import COLORS, MARKERS, LINE_STYLES

class MutualInformationVisualizer:
    """互信息可视化器，用于比较模型间互信息值的变化"""
    
    def __init__(self, model_visualizer):
        """
        初始化互信息可视化器
        
        Args:
            model_visualizer: ModelVisualizer实例，提供数据访问方法
        """
        self.model_visualizer = model_visualizer
        self.base_dir = model_visualizer.base_dir
        self.model_dirs = model_visualizer.model_dirs
        self.model_names = model_visualizer.model_names
    
    def plot(self, save_path=None):
        """
        比较不同模型的互信息值
        
        Args:
            save_path: 保存图表的路径，如果为None则显示图表
            
        Returns:
            matplotlib.figure.Figure: 生成的图表对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        has_data = False
        max_mi_value = 0
        min_mi_value = float('inf')
        max_epochs = 0
        
        # 遍历所有模型并绘制互信息
        for i, model_dir in enumerate(self.model_dirs):
            epochs, _, _, mi_values = self.model_visualizer._load_log(model_dir)
            
            if mi_values and len(mi_values) > 0:
                has_data = True
                
                # 检查数据维度是否匹配
                if len(epochs) != len(mi_values):
                    print(f"Warning: MI data dimensions mismatch for {model_dir}. Epochs: {len(epochs)}, MI values: {len(mi_values)}")
                    # 调整到相同长度
                    min_len = min(len(epochs), len(mi_values))
                    epochs = epochs[:min_len]
                    mi_values = mi_values[:min_len]
                
                # 使用前面的数据绘制基础趋势线，最后一个点标记为最终MI
                if len(epochs) > 0 and len(epochs) == len(mi_values):
                    ax.plot(epochs, mi_values, 
                          marker=MARKERS[i % len(MARKERS)],
                          linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                          color=COLORS[i % len(COLORS)],
                          label=self.model_names[i],
                          linewidth=1.5,
                          markersize=5)
                    
                    # 标记最终MI值
                    final_epoch = epochs[-1]
                    final_mi = mi_values[-1]
                    ax.scatter(final_epoch, final_mi, 
                             s=100, 
                             color=COLORS[i % len(COLORS)], 
                             edgecolor='black',
                             zorder=10,
                             marker='o')
                    
                    # 在最终点添加标签
                    ax.text(final_epoch, final_mi, f" {final_mi:.4f}", 
                          fontsize=9, 
                          verticalalignment='center')
                else:
                    # 对于只有一个MI值的情况，显示为水平线
                    final_mi = mi_values[-1]
                    ax.axhline(y=final_mi, 
                             linestyle='--',
                             color=COLORS[i % len(COLORS)],
                             label=f"{self.model_names[i]} (MI={final_mi:.4f})")
                
                # 更新最大/最小MI值，用于设置y轴范围
                if mi_values:
                    max_mi_value = max(max_mi_value, max(mi_values))
                    min_mi_value = min(min_mi_value, min(mi_values))
                
                # 更新最大epoch数，用于设置x轴范围
                if epochs and len(epochs) > 0:
                    max_epochs = max(max_epochs, max(epochs))
        
        # 如果没有数据，显示消息
        if not has_data:
            ax.text(0.5, 0.5, "No Mutual Information Data Available", 
                  fontsize=12, fontweight='bold',
                  horizontalalignment='center',
                  verticalalignment='center',
                  transform=ax.transAxes)
        else:
            # 添加标题和标签
            ax.set_title("Mutual Information Comparison", fontsize=14)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Mutual Information (bits)", fontsize=12)
            
            # 设置较好的y轴范围
            if min_mi_value != float('inf') and max_mi_value > 0:
                y_margin = (max_mi_value - min_mi_value) * 0.1 if max_mi_value > min_mi_value else max_mi_value * 0.1
                ax.set_ylim(max(0, min_mi_value - y_margin), max_mi_value + y_margin)
            
            # 确保x轴是整数
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            # 添加网格和图例
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='best', frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 保存高分辨率位图
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            
            # 也保存矢量格式
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight')
                print(f"Mutual information comparison saved to {save_path} and {vector_path}")
            else:
                print(f"Mutual information comparison saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        return fig


class MutualInformationOverTimeVisualizer:
    """互信息随时间变化可视化器，用于展示模型训练过程中互信息的变化趋势"""
    
    def __init__(self, model_visualizer):
        """
        初始化互信息随时间变化可视化器
        
        Args:
            model_visualizer: ModelVisualizer实例，提供数据访问方法
        """
        self.model_visualizer = model_visualizer
        self.base_dir = model_visualizer.base_dir
        self.model_dirs = model_visualizer.model_dirs
        self.model_names = model_visualizer.model_names
    
    def plot(self, save_path=None, log_scale=False, smoothing=0.0):
        """
        绘制互信息随时间变化的对比图
        
        Args:
            save_path: 保存图表的路径，如果为None则显示图表
            log_scale: 是否使用对数刻度
            smoothing: 平滑因子 (0-1)，0表示无平滑
            
        Returns:
            matplotlib.figure.Figure: 生成的图表对象
        """
        fig, ax = plt.subplots(figsize=(8, 5.5))
        
        for i, model_dir in enumerate(self.model_dirs):
            epochs, _, _, mi = self.model_visualizer._load_log(model_dir)
            
            if not mi or len(mi) == 0:
                continue
            
            # 检查数据维度是否匹配
            if len(epochs) != len(mi):
                print(f"Warning: MI data dimensions mismatch for {model_dir}. Epochs: {len(epochs)}, MI values: {len(mi)}")
                # 调整到相同长度
                min_len = min(len(epochs), len(mi))
                epochs = epochs[:min_len]
                mi = mi[:min_len]
                
            # 应用平滑处理（如果需要）
            if smoothing > 0 and len(mi) > 1:
                smooth_mi = [mi[0]]
                for j in range(1, len(mi)):
                    smooth_mi.append(smoothing * smooth_mi[-1] + (1 - smoothing) * mi[j])
                mi = smooth_mi
            
            # 绘制具有独特样式的线条
            ax.plot(epochs, mi, 
                   label=self.model_names[i],
                   color=COLORS[i % len(COLORS)], 
                   marker=MARKERS[i % len(MARKERS)],
                   markersize=5,
                   markevery=max(1, len(epochs)//10),  # 每10个点放置一个标记
                   linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                   linewidth=1.8,
                   alpha=0.9)
        
        # 设置描述性标签
        ax.set_xlabel('训练轮次')
        ax.set_ylabel('互信息 (nats)')
        
        if log_scale and ax.get_ylim()[0] > 0:
            ax.set_yscale('log')
            
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 添加图例
        ax.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True)
        
        # 设置合适的轴限制
        ax.set_xlim(left=0)
        if not log_scale:
            ax.set_ylim(bottom=0)
        
        # 添加标题
        ax.set_title('互信息随训练轮次的变化')
        
        plt.tight_layout()
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 保存高分辨率位图
            plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
            
            # 也保存矢量格式
            if save_path.endswith('.png'):
                vector_path = save_path.replace('.png', '.pdf')
                plt.savefig(vector_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
                print(f"互信息随时间变化图已保存至 {save_path} 和 {vector_path}")
            else:
                print(f"互信息随时间变化图已保存至 {save_path}")
        else:
            plt.show()
        
        plt.close()
        return fig 