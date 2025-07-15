#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
样式工具模块 - 为可视化提供符合国际学术发表标准的专业样式设置
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cycler

# 符合IEEE/ACM国际会议标准的颜色方案，基于Color Brewer配色
# 保证色盲友好、打印友好且在黑白打印时可区分
COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442']
GRAY_COLORS = ['#252525', '#636363', '#969696', '#bdbdbd', '#d9d9d9']

# 科学图表标准的标记样式，确保即使在黑白打印时也能区分
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '<', '>']
LINE_STYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]

# 国际学术期刊标准图表参数
DEFAULT_DPI = 600           # 高分辨率 (>=300dpi为期刊标准)
DEFAULT_FIG_WIDTH = 7.5     # 单栏期刊图表宽度（英寸）
DEFAULT_FIG_HEIGHT = 5.5    # 适合学术出版的高度
DEFAULT_COLUMN_WIDTH = 3.5  # IEEE单栏宽度（英寸）
DEFAULT_TWO_COLUMN_WIDTH = 7.16  # IEEE双栏宽度（英寸）

# 数据集特定参数，支持常见的视觉数据集
DATASET_PARAMS = {
    'mnist': {'img_shape': (28, 28), 'channels': 1, 'color_map': 'gray'},
    'binary_mnist': {'img_shape': (28, 28), 'channels': 1, 'color_map': 'gray'},
    'binarymnist': {'img_shape': (28, 28), 'channels': 1, 'color_map': 'gray'},
    'cifar10': {'img_shape': (32, 32), 'channels': 3, 'color_map': None},
    'svhn': {'img_shape': (32, 32), 'channels': 3, 'color_map': None},
    'omniglot': {'img_shape': (105, 105), 'channels': 1, 'color_map': 'gray'},
    'celeba': {'img_shape': (64, 64), 'channels': 3, 'color_map': None}
}

def set_publication_style(dpi=DEFAULT_DPI, use_latex=False):
    """
    设置符合国际学术期刊发表标准的matplotlib样式
    
    Args:
        dpi: 分辨率, 默认600dpi (IEEE推荐)
        use_latex: 是否使用LaTeX排版，默认False
    """
    # 使用标准学术字体
    if use_latex:
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
            'text.latex.preamble': r'\usepackage{amsmath,amssymb,amsfonts,amsthm}' 
        })
    else:
        plt.rcParams.update({
            'text.usetex': False,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
            'mathtext.fontset': 'cm'  # 数学文本使用Computer Modern
        })
    
    # IEEE标准图表字体大小
    plt.rcParams.update({
        'font.size': 10,        # 基本字体大小
        'axes.titlesize': 11,   # 标题字体大小
        'axes.labelsize': 10,   # 坐标轴标签字体大小
        'xtick.labelsize': 9,   # X轴刻度标签字体大小
        'ytick.labelsize': 9,   # Y轴刻度标签字体大小
        'legend.fontsize': 9,   # 图例字体大小
        'figure.titlesize': 12  # 图表主标题字体大小
    })
    
    # 专业线条和标记设置
    plt.rcParams.update({
        'lines.linewidth': 1.5,       # 线宽
        'lines.markersize': 5,        # 标记大小
        'lines.markeredgewidth': 1.0, # 标记边缘宽度
        'axes.linewidth': 0.8,        # 坐标轴线宽
        'xtick.major.width': 0.8,     # X轴主刻度宽度
        'ytick.major.width': 0.8,     # Y轴主刻度宽度
        'xtick.minor.width': 0.6,     # X轴次刻度宽度
        'ytick.minor.width': 0.6,     # Y轴次刻度宽度
        'xtick.major.size': 3.5,      # X轴主刻度长度
        'ytick.major.size': 3.5,      # Y轴主刻度长度
        'xtick.minor.size': 2.0,      # X轴次刻度长度
        'ytick.minor.size': 2.0,      # Y轴次刻度长度
        'xtick.direction': 'in',      # X轴刻度朝内（学术标准）
        'ytick.direction': 'in'       # Y轴刻度朝内（学术标准）
    })
    
    # 专业网格设置
    plt.rcParams.update({
        'axes.grid': True,          # 使用网格
        'grid.alpha': 0.3,          # 网格透明度
        'grid.linestyle': '--',     # 网格线样式
        'grid.linewidth': 0.5       # 网格线宽度
    })
    
    # 图表布局
    plt.rcParams.update({
        'figure.constrained_layout.use': True,   # 更好的自动布局
        'figure.autolayout': False,              # 禁用可能冲突的旧布局
        'savefig.bbox': 'tight',                 # 保存时紧凑布局
        'savefig.pad_inches': 0.1                # 边缘留白
    })
    
    # 高质量图像输出设置
    plt.rcParams.update({
        'savefig.dpi': dpi,                # 保存图表的DPI
        'figure.dpi': 100,                 # 屏幕显示DPI
        'savefig.format': 'pdf',           # 默认保存格式为PDF（矢量格式）
        'pdf.fonttype': 42,                # 确保PDF中的字体为嵌入字体
        'ps.fonttype': 42                  # 确保PS/EPS中的字体为嵌入字体
    })
    
    # 专业色彩循环设置，确保多个数据系列可区分
    plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', COLORS)

def get_figure_params(width_type='single'):
    """
    获取不同发表场景的推荐图表参数
    
    Args:
        width_type: 字符串, 可选值包括:
                   'single' - 单栏格式 (默认, 适用于IEEE/ACM Transactions)
                   'double' - 双栏格式 (适用于IEEE/ACM会议)
                   'full' - 全页宽度 (适用于演示文稿)
                   
    Returns:
        dict: 包含推荐的宽度和高度参数
    """
    if width_type == 'single':
        return {
            'figsize': (DEFAULT_COLUMN_WIDTH, DEFAULT_COLUMN_WIDTH * 0.75),
            'dpi': DEFAULT_DPI
        }
    elif width_type == 'double':
        return {
            'figsize': (DEFAULT_TWO_COLUMN_WIDTH, DEFAULT_TWO_COLUMN_WIDTH * 0.5),
            'dpi': DEFAULT_DPI
        }
    elif width_type == 'full':
        return {
            'figsize': (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT),
            'dpi': DEFAULT_DPI
        }
    else:
        return {
            'figsize': (DEFAULT_COLUMN_WIDTH, DEFAULT_COLUMN_WIDTH * 0.75),
            'dpi': DEFAULT_DPI
        }

def create_publication_figure(width_type='single', aspect_ratio=0.75):
    """
    创建符合国际发表标准的图表
    
    Args:
        width_type: 字符串, 图表宽度类型 ('single', 'double', 'full')
        aspect_ratio: 高宽比, 默认0.75 (4:3)
        
    Returns:
        figure, axes: matplotlib图表和轴对象
    """
    params = get_figure_params(width_type)
    width = params['figsize'][0]
    height = width * aspect_ratio
    
    fig, ax = plt.subplots(figsize=(width, height), dpi=params['dpi'])
    
    # 应用通用风格设置
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ax.tick_params(which='both', direction='in')
    ax.tick_params(which='major', length=3.5)
    ax.tick_params(which='minor', length=2.0)
    
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    return fig, ax

def format_axis_scientific(ax, axis='y', precision=1):
    """
    将坐标轴格式化为科学计数法
    
    Args:
        ax: matplotlib轴对象
        axis: 要格式化的轴 ('x', 'y' 或 'both')
        precision: 小数点后位数
    """
    formatter = mpl.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 4))  # 小于10^-3或大于10^4时使用科学计数法
    
    if axis == 'x' or axis == 'both':
        ax.xaxis.set_major_formatter(formatter)
    if axis == 'y' or axis == 'both':
        ax.yaxis.set_major_formatter(formatter)

def add_subplot_label(ax, label, loc='upper left', offset=(0.05, -0.05)):
    """
    为子图添加标签 (a), (b), (c)等，符合学术出版标准
    
    Args:
        ax: matplotlib轴对象
        label: 标签文本, 如 '(a)', '(b)'
        loc: 标签位置
        offset: 位置偏移量, 相对于坐标范围的比例
    """
    pos = ax.get_position()
    if loc == 'upper left':
        x = pos.x0 + offset[0]
        y = pos.y1 + offset[1]
    elif loc == 'upper right':
        x = pos.x1 + offset[0]
        y = pos.y1 + offset[1]
    elif loc == 'lower left':
        x = pos.x0 + offset[0]
        y = pos.y0 + offset[1]
    elif loc == 'lower right':
        x = pos.x1 + offset[0]
        y = pos.y0 + offset[1]
    
    fig = ax.get_figure()
    fig.text(x, y, label, fontsize=11, fontweight='bold', 
             ha='center', va='center')

def create_colormap(cmap_name='viridis', reverse=False):
    """
    创建适合学术发表的色图
    
    Args:
        cmap_name: 基础色图名称
        reverse: 是否反转色图
        
    Returns:
        matplotlib色图对象
    """
    cmap = plt.cm.get_cmap(cmap_name)
    if reverse:
        cmap = cmap.reversed()
    return cmap

def save_publication_figure(fig, filename, formats=None):
    """
    以多种格式保存符合出版标准的图表
    
    Args:
        fig: matplotlib图表对象
        filename: 基础文件名(不含扩展名)
        formats: 保存格式列表，如['pdf', 'png', 'eps']
    """
    if formats is None:
        formats = ['pdf', 'png']
        
    for fmt in formats:
        output_file = f"{filename}.{fmt}"
        if fmt == 'pdf' or fmt == 'eps':
            fig.savefig(output_file, format=fmt, bbox_inches='tight', 
                      pad_inches=0.1, dpi=DEFAULT_DPI)
        else:
            fig.savefig(output_file, format=fmt, bbox_inches='tight', 
                      pad_inches=0.1, dpi=DEFAULT_DPI)
    
    print(f"保存图表到: {filename}.{{{','.join(formats)}}}")
