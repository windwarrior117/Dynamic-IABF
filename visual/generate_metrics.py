#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成metrics.pkl文件工具

此脚本用于从模型的重建结果和日志文件中提取评估指标，
并将它们保存为metrics.pkl文件，以便可视化工具使用。
"""

import os
import pickle
import numpy as np
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def extract_metrics_from_log(log_file):
    """从日志文件提取测试指标"""
    metrics = {}
    
    if not os.path.exists(log_file):
        print(f"警告: 日志文件不存在: {log_file}")
        return metrics
    
    with open(log_file, 'r') as f:
        for line in f:
            # 提取L2测试损失
            if "L2 squared test loss (per image)" in line:
                try:
                    value = float(line.split(':')[-1].strip())
                    metrics['l2_squared_loss'] = value
                except:
                    pass
                    
            elif "L2 test loss (per image)" in line:
                try:
                    value = float(line.split(':')[-1].strip())
                    metrics['l2_loss'] = value
                except:
                    pass
                    
            elif "L2 squared test loss (per pixel)" in line:
                try:
                    value = float(line.split(':')[-1].strip())
                    metrics['l2_squared_pixel_loss'] = value
                except:
                    pass
                    
            elif "L2 test loss (per pixel)" in line:
                try:
                    value = float(line.split(':')[-1].strip())
                    metrics['l2_pixel_loss'] = value
                except:
                    pass
            
            # 提取互信息
            elif "mutual information" in line.lower():
                try:
                    value = float(line.split()[-1].strip())
                    metrics['mi'] = value
                except:
                    pass
                    
    return metrics

def compute_reconstruction_metrics(reconstr_file, test_data_file=None):
    """计算重建质量指标（PSNR, SSIM, MSE, MAE）"""
    metrics = {}
    
    if not os.path.exists(reconstr_file):
        print(f"警告: 重建文件不存在: {reconstr_file}")
        return metrics
    
    try:
        with open(reconstr_file, 'rb') as f:
            reconstructions = pickle.load(f)
            
        # 检查重建是否为numpy数组
        if not isinstance(reconstructions, np.ndarray):
            print(f"警告: 重建数据不是numpy数组: {type(reconstructions)}")
            return metrics
            
        # 如果有原始测试数据，计算与之相关的指标
        if test_data_file and os.path.exists(test_data_file):
            with open(test_data_file, 'rb') as f:
                test_data = pickle.load(f)
                
            # 检查数据类型
            if isinstance(test_data, tuple) and len(test_data) >= 2:
                # 处理返回(flat_data, image_data)的情况
                originals = test_data[0]
            else:
                originals = test_data
                
            # 确保originals是numpy数组
            if not isinstance(originals, np.ndarray):
                originals = np.array(originals)
                
            # 检查是否有足够的样本
            if len(originals) < len(reconstructions):
                print(f"警告: 测试样本数量不足 ({len(originals)}) < ({len(reconstructions)})")
                originals = originals[:min(len(originals), len(reconstructions))]
                reconstructions = reconstructions[:min(len(originals), len(reconstructions))]
            else:
                originals = originals[:len(reconstructions)]
                
            # 确保值在[0,1]范围内
            if originals.max() > 1.0:
                originals = originals / 255.0
            if reconstructions.max() > 1.0:
                reconstructions = reconstructions / 255.0
                
            # 计算MSE和MAE
            mse = np.mean(np.square(originals - reconstructions))
            mae = np.mean(np.abs(originals - reconstructions))
            metrics['mse'] = float(mse)
            metrics['mae'] = float(mae)
            
            # 计算PSNR和SSIM
            # 需要将向量重塑为图像
            if originals.ndim == 2 and originals.shape[1] in [784, 3072, 12288]:
                # MNIST (28x28), CIFAR (32x32x3), CelebA (64x64x3)
                psnr_vals = []
                ssim_vals = []
                
                for i in range(len(originals)):
                    orig = originals[i]
                    recon = reconstructions[i]
                    
                    # 重塑为图像
                    if orig.size == 784:  # MNIST
                        orig = orig.reshape(28, 28)
                        recon = recon.reshape(28, 28)
                        psnr_val = psnr(orig, recon, data_range=1.0)
                        ssim_val = ssim(orig, recon, data_range=1.0)
                    elif orig.size == 3072:  # CIFAR
                        orig = orig.reshape(32, 32, 3)
                        recon = recon.reshape(32, 32, 3)
                        psnr_val = psnr(orig, recon, data_range=1.0)
                        ssim_val = ssim(orig, recon, data_range=1.0, channel_axis=2)
                    elif orig.size == 12288:  # CelebA
                        orig = orig.reshape(64, 64, 3)
                        recon = recon.reshape(64, 64, 3)
                        psnr_val = psnr(orig, recon, data_range=1.0)
                        ssim_val = ssim(orig, recon, data_range=1.0, channel_axis=2)
                    else:
                        continue
                        
                    psnr_vals.append(psnr_val)
                    ssim_vals.append(ssim_val)
                
                if psnr_vals:
                    metrics['psnr'] = float(np.mean(psnr_vals))
                if ssim_vals:
                    metrics['ssim'] = float(np.mean(ssim_vals))
    except Exception as e:
        print(f"计算重建指标时出错: {str(e)}")
        
    return metrics

def extract_distribution_from_log(log_file):
    """从日志文件中提取分布数据"""
    if not os.path.exists(log_file):
        return None
        
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # 查找分布数据的正则表达式模式
        import re
        pattern = r"distribution_list\] (\[.*\])"
        match = re.search(pattern, content)
        if match:
            # 提取和解析数组字符串
            data_str = match.group(1)
            # 将字符串转换为Python列表
            distribution = eval(data_str)
            return np.array(distribution)
    except Exception as e:
        print(f"提取分布数据时出错: {str(e)}")
        
    return None

def generate_metrics_pkl(model_dir, test_data_file=None, overwrite=False):
    """为指定模型目录生成metrics.pkl文件"""
    metrics_file = os.path.join(model_dir, 'metrics.pkl')
    
    # 如果文件已存在且不覆盖，则跳过
    if os.path.exists(metrics_file) and not overwrite:
        print(f"metrics.pkl已经存在: {metrics_file}")
        return False
        
    # 收集指标
    metrics = {}
    
    # 从日志文件提取指标
    log_file = os.path.join(model_dir, 'log.txt')
    log_metrics = extract_metrics_from_log(log_file)
    metrics.update(log_metrics)
    
    # 从重建结果计算指标
    reconstr_file = os.path.join(model_dir, 'reconstr.pkl')
    recon_metrics = compute_reconstruction_metrics(reconstr_file, test_data_file)
    metrics.update(recon_metrics)
    
    # 从日志中提取分布数据
    distribution = extract_distribution_from_log(log_file)
    if distribution is not None:
        metrics['distribution'] = distribution
        # 同时保存为npy文件以便可视化使用
        np.save(os.path.join(model_dir, 'distribution.npy'), distribution)
    
    # 保存指标
    if metrics:
        with open(metrics_file, 'wb') as f:
            pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)
        print(f"成功生成metrics.pkl: {metrics_file}")
        print(f"指标内容: {metrics}")
        return True
    else:
        print(f"警告: 未找到任何指标，未生成文件: {metrics_file}")
        return False

def main():
    parser = argparse.ArgumentParser(description='为模型目录生成metrics.pkl文件')
    parser.add_argument('--results_dir', type=str, default='./results', 
                        help='结果根目录')
    parser.add_argument('--model_type', type=str, default=None, 
                        help='要处理的模型类型，如DynamicIABF/IABF/NECST/UAE，不指定则处理所有类型')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='数据集名称')
    parser.add_argument('--test_data', type=str, default=None,
                        help='测试数据文件路径，用于计算重建指标')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的metrics.pkl文件')
    
    args = parser.parse_args()
    
    # 如果未指定测试数据文件，使用默认路径
    if args.test_data is None:
        args.test_data = f'./data/{args.dataset}_test_data.pkl'
        
    # 查找所有模型目录
    if args.model_type:
        base_dir = os.path.join(args.results_dir, args.model_type)
        if os.path.exists(base_dir):
            model_types = [args.model_type]
        else:
            print(f"错误: 模型类型目录不存在: {base_dir}")
            return
    else:
        # 获取所有模型类型
        model_types = []
        for d in os.listdir(args.results_dir):
            if os.path.isdir(os.path.join(args.results_dir, d)):
                model_types.append(d)
    
    # 处理每个模型目录
    for model_type in model_types:
        type_dir = os.path.join(args.results_dir, model_type)
        
        # 获取数据集目录
        dataset_dirs = []
        for d in os.listdir(type_dir):
            dataset_dir = os.path.join(type_dir, d)
            if os.path.isdir(dataset_dir):
                if args.dataset and d.lower() != args.dataset.lower():
                    continue
                dataset_dirs.append(dataset_dir)
        
        for dataset_dir in dataset_dirs:
            # 获取具体实验目录
            exp_dirs = []
            for d in os.listdir(dataset_dir):
                exp_dir = os.path.join(dataset_dir, d)
                if os.path.isdir(exp_dir):
                    exp_dirs.append(exp_dir)
            
            for exp_dir in exp_dirs:
                print(f"处理目录: {exp_dir}")
                # 检查是否有reconstr.pkl
                if os.path.exists(os.path.join(exp_dir, 'reconstr.pkl')):
                    generate_metrics_pkl(exp_dir, args.test_data, args.overwrite)
                else:
                    print(f"跳过目录(无reconstr.pkl): {exp_dir}")

if __name__ == "__main__":
    main() 