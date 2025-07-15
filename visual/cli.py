#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行接口模块 - 处理命令行参数并调用可视化功能
"""

import os
import argparse
from .data_utils import extract_test_data
from .model_utils import find_model_dirs
from .core import ModelVisualizer

def main_cli():
    """处理命令行参数并执行可视化操作"""
    parser = argparse.ArgumentParser(description="Model Performance Visualization Tool")
    parser.add_argument("--base_dir", type=str, default="./results/", 
                        help="Base directory for result files")
    parser.add_argument("--model_dirs", type=str, nargs="+", 
                        help="List of model result subdirectories")
    parser.add_argument("--model_names", type=str, nargs="+", 
                        help="Model names used in legends")
    parser.add_argument("--model_types", type=str, nargs="+", 
                        help="Model types to display, e.g. DynamicIABF, IABF, NECST, UAE")
    parser.add_argument("--output_dir", type=str, default="./plots/", 
                        help="Output directory for saving images")
    parser.add_argument("--auto_discover", action="store_true", 
                        help="Automatically discover model directories")
    parser.add_argument("--plot_type", type=str, 
                        choices=["all", "learning", "mi", "distribution", "reconstruction", 
                                "metrics", "noise", "table"], 
                        default="all", help="Type of plot to generate")
    parser.add_argument("--extract_data", action="store_true",
                        help="Force extraction and update of test data")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of test samples to extract")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "binary_mnist", "binarymnist", "cifar10", "svhn", "omniglot", "celeba"],
                        help="Dataset type for visualization")
    
    args = parser.parse_args()
    
    # 如果指定了数据提取，首先提取测试数据
    if args.extract_data:
        extract_test_data(args.dataset, num_samples=args.num_samples, force_update=True)
    else:
        # 确保测试数据存在
        extract_test_data(args.dataset, num_samples=args.num_samples)
    
    if args.auto_discover:
        # 如果指定了模型类型，只显示这些类型
        selected_types = args.model_types if args.model_types else ["DynamicIABF", "IABF", "NECST", "UAE"]
        model_dirs, model_names = find_model_dirs(args.base_dir, selected_types)
        if not model_dirs:
            print("No model result directories found")
            return
        print(f"Found models: {', '.join(model_names)}")
    else:
        model_dirs = args.model_dirs if args.model_dirs else []
        model_names = args.model_names if args.model_names else ["Model " + str(i+1) for i in range(len(model_dirs))]
    
    # 创建可视化器
    visualizer = ModelVisualizer(args.base_dir, model_dirs, model_names, args.dataset)
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 根据指定的类型生成图表
    if args.plot_type == "all":
        try:
            visualizer.plot_all(args.output_dir)
        except Exception as e:
            print(f"Error generating some plots: {str(e)}")
    elif args.plot_type == "learning":
        save_path = os.path.join(args.output_dir, "learning_curves.png")
        visualizer.plot_learning_curves(save_path)
    elif args.plot_type == "mi":
        save_path = os.path.join(args.output_dir, "mutual_information.png")
        visualizer.plot_mutual_information(save_path)
    elif args.plot_type == "distribution":
        save_path = os.path.join(args.output_dir, "distribution_comparison.png")
        visualizer.plot_distribution_comparison(save_path)
    elif args.plot_type == "reconstruction":
        save_path = os.path.join(args.output_dir, "reconstruction_samples.png")
        visualizer.plot_reconstruction_samples(save_path=save_path)
    elif args.plot_type == "metrics":
        save_path = os.path.join(args.output_dir, "test_metrics.png")
        visualizer.plot_test_metrics(save_path)
    elif args.plot_type == "noise":
        save_path = os.path.join(args.output_dir, "noise_analysis.png")
        visualizer.plot_noise_analysis(save_path)
    elif args.plot_type == "table":
        csv_path = os.path.join(args.output_dir, "reconstruction_errors.csv")
        save_path = os.path.join(args.output_dir, "reconstruction_errors_table.png")
        visualizer.plot_reconstruction_errors_table(save_path, csv_path)