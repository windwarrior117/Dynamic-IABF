# DynamicIABF: 具有自适应噪声和渐进式训练的信息最大化神经联合信源信道编码

---

## 📋 项目概览

DynamicIABF是对IABF（Infomax Adversarial Bit Flip）的创新改进，通过引入自适应噪声水平调整和渐进式训练策略，进一步提高了在噪声信道条件下的图像传输质量。本项目利用深度学习技术实现了一种智能通信系统，能够根据训练过程中的反馈动态调整编码策略，显著增强抗噪性能。

### **主要特点**
- ✅ 自适应噪声水平调整机制，实时响应训练反馈
- ✅ 渐进式训练策略，从简单到复杂逐步适应噪声环境
- ✅ 基于对抗性比特翻转的信息最大化编码学习
- ✅ 支持多种信道模型（BSC/BEC）和数据集
- ✅ 与传统方法（IABF、NECST和UAE）相比具有更强的适应性和抗噪能力

### **理论基础**
DynamicIABF基于两个核心理念：
1. **适应性学习**：通信系统应当能够根据信道条件动态调整策略
2. **渐进式复杂度**：从简单（低噪声）环境开始训练，逐步增加难度（高噪声）

这种方法模拟了人类学习复杂任务的方式，并在实验中显示出显著的性能提升。

---

## 目录

- [数据集准备](#-数据集准备)
- [命令行参数](#%EF%B8%8F-命令行参数)
- [使用示例](#-使用示例)
- [模型比较](#-模型比较)
- [自动化对照训练脚本](#自动化对照训练脚本)
- [完整训练流程教程](#完整训练流程教程)
- [常见问题](#常见问题)
- [更新日志](#更新日志)

---

## 🔍 数据集准备

本项目使用TensorFlow的[TFRecords](https://www.tensorflow.org/tutorials/load_data/tf_records)格式处理数据。数据准备流程：

### 1. **下载原始数据**
```bash
# MNIST/BinaryMNIST数据集
python3 data_setup/download.py mnist
# 或下载二值化MNIST
python3 data_setup/download.py BinaryMNIST

# CelebA数据集
python3 data_setup/celebA_download.py

# CIFAR10数据集
python3 data_setup/generate_cifar10_tfrecords.py
```
**注意**：Omniglot和SVHN需要单独下载

### 2. **转换数据格式**
```bash
# 转换CelebA为HDF5格式
python3 data_setup/convert_celebA_h5.py

# 转换Omniglot为HDF5格式
python3 data_setup/convert_omniglot_h5.py

# 生成随机二进制数据
python3 data_setup/gen_random_bits.py
```

### 3. **生成TFRecords**
```bash
# 转换MNIST为TFRecords
python3 data_setup/convert_to_records.py --dataset=mnist

# 转换BinaryMNIST为TFRecords
python3 data_setup/convert_to_records.py --dataset=BinaryMNIST

# 其他数据集类似
python3 data_setup/convert_to_records.py --dataset=<数据集名称>
```

---

## ⚙️ 命令行参数

### 通用参数

训练模型时可用的主要命令行参数：

| 参数 | 说明 | 默认值 |
|:------:|:------:|:-------:|
| `--exp_id` | 实验标识符 | 必填 |
| `--flip_samples` | 对抗训练中翻转的位数 | 0 |
| `--miw` | 互信息项的权重 | 0.0 |
| `--datasource` | 数据源类型 | "mnist" |
| `--is_binary` | 数据是否为二进制 | False |
| `--vimco_samples` | VIMCO使用的样本数量 | 0 |
| `--channel_model` | 信道模型类型 | "bsc" |
| `--noise` | 训练时的信道噪声水平 | 0.1 |
| `--test_noise` | 测试时的信道噪声水平 | 0.1 |
| `--n_bits` | 编码的比特数 | 100 |
| `--n_epochs` | 训练轮数 | 10 |
| `--batch_size` | 小批量大小 | 100 |
| `--lr` | 优化器学习率 | 0.001 |
| `--optimizer` | 优化器类型 | "adam" |
| `--dec_arch` | 解码器架构 | "500,500" |
| `--enc_arch` | 编码器架构 | "500,500" |
| `--reg_param` | 编码器的正则化参数 | 0.0001 |
| `--datadir` | 数据目录 | "./data" |
| `--logdir` | 日志目录 | "./logs" |
| `--outdir` | 输出目录 | "./results" |

### DynamicIABF特有参数

DynamicIABF模型引入了以下额外参数来控制自适应噪声和渐进式训练：

| 参数 | 说明 | 默认值 |
|:------:|:------:|:-------:|
| `--use_dynamic_model` | 是否启用DynamicIABF模型 | False |
| `--adaptive_noise` | 是否启用自适应噪声水平调整 | False |
| `--progressive_training` | 是否启用渐进式训练策略 | False |
| `--noise_min` | 渐进式训练的初始/最小噪声水平 | 0.01 |
| `--noise_max` | 渐进式训练的最大噪声水平 | 0.3 |
| `--noise_adapt_rate` | 自适应噪声调整率 | 0.05 |

#### **参数详解**

- **自适应噪声机制**：通过`--adaptive_noise=True`启用，系统会根据训练过程中的损失变化动态调整噪声水平。当模型在当前噪声水平下表现良好（损失减小）时，系统会适度增加噪声水平，以挑战模型并提高其鲁棒性；反之则降低噪声水平，使模型能够稳定学习。

- **渐进式训练策略**：通过`--progressive_training=True`启用，系统会从较低的噪声水平（`--noise_min`）开始训练，随着训练的进行，噪声水平会逐渐增加到设定的最大值（`--noise_max`）。这种方法模拟了"由易到难"的学习过程，使模型能够更有效地适应高噪声环境。

- **噪声调整率**：通过`--noise_adapt_rate`控制噪声水平变化的幅度。较大的值会导致噪声水平变化更加激进，而较小的值则使变化更为平缓。

### IABF特有参数

IABF模型的特有参数侧重于对抗训练和互信息优化：

| 参数 | 说明 | 默认值 |
|:------:|:------:|:-------:|
| `--flip_samples` | 对抗训练中翻转的位数 | 0 |
| `--miw` | 互信息项的权重 | 0.0 |
| `--noise` | 训练时的信道噪声水平 | 0.1 |
| `--vimco_samples` | VIMCO互信息估计使用的样本数量 | 0 |

#### **参数详解**

- **对抗位翻转数量**：通过`--flip_samples`设置，控制在对抗训练过程中翻转多少位。对于100位编码，通常设置为5-10位较为合适。该参数直接影响对抗训练的强度，值太大可能导致训练困难，太小则效果不明显。

- **互信息权重**：通过`--miw`设置，控制互信息最大化项在损失函数中的权重。这是一个关键超参数，通常取较小的值（如1e-7）。值太大可能导致训练不稳定，太小则互信息约束效果不明显。

- **固定噪声水平**：通过`--noise`设置训练时使用的固定信道噪声水平。不同于DynamicIABF的动态调整，IABF使用恒定的噪声水平进行训练，因此选择合适的噪声水平对模型性能至关重要。

- **VIMCO样本数量**：通过`--vimco_samples`设置，用于互信息估计的变分推断样本数。增加样本数可提高互信息估计的准确性，但也会增加计算开销。

### NECST特有参数

NECST模型专注于信道编码，不使用对抗训练和互信息优化：

| 参数 | 说明 | 默认值 |
|:------:|:------:|:-------:|
| `--noise` | 训练时的信道噪声水平 | 0.1 |
| `--channel_model` | 信道模型类型 | "bsc" |

#### **参数详解**

- **噪声水平**：通过`--noise`设置训练时的固定信道噪声水平。NECST依赖于随机信道噪声来学习鲁棒的编码，因此噪声水平应设置为与目标应用场景相符的值。

- **信道模型**：通过`--channel_model`选择使用哪种信道模型。NECST支持二进制对称信道(BSC)和二进制擦除信道(BEC)，不同信道模型会产生不同类型的噪声，进而影响编码学习。

### UAE特有参数

UAE作为基线模型，不使用噪声、对抗训练或互信息优化，主要使用标准自编码器参数：

| 参数 | 说明 | 默认值 |
|:------:|:------:|:-------:|
| `--reg_param` | 编码器的正则化参数 | 0.0001 |
| `--enc_arch` | 编码器架构 | "500,500" |
| `--dec_arch` | 解码器架构 | "500,500" |

#### **参数详解**

- **正则化参数**：通过`--reg_param`控制编码器的正则化强度。适当的正则化有助于防止过拟合并提高编码的泛化能力。对于UAE，由于没有噪声和特殊约束，合适的正则化尤为重要。

- **网络架构**：通过`--enc_arch`和`--dec_arch`设置编码器和解码器的网络结构。格式为以逗号分隔的隐藏层神经元数量，如"500,500"表示两个隐藏层，每层500个神经元。对于UAE，合适的网络容量对于获得高质量重建至关重要。

---

## 🚀 使用示例

### 基本示例：训练DynamicIABF模型

以下是训练具有自适应噪声和渐进式训练功能的DynamicIABF模型的完整流程：

```bash
# 步骤1：下载MNIST数据集
python3 data_setup/download.py mnist

# 步骤2：生成对应的tfrecords文件
python3 data_setup/convert_to_records.py --dataset=mnist

# 步骤3：训练DynamicIABF模型（完整版，包含自适应噪声和渐进式训练）
python3 main.py --exp_id="dynamic_iabf_test" --use_dynamic_model=True --adaptive_noise=True \
  --progressive_training=True --noise_min=0.01 --noise_max=0.3 --noise_adapt_rate=0.05 \
  --flip_samples=7 --miw=0.0000001 --noise=0.1 --test_noise=0.1 --datadir=./data \
  --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100 \
  --lr=0.001 --optimizer=adam --is_binary=False --dec_arch="500,500" --enc_arch="500,500" \
  --reg_param=0.0001
```

DynamicIABF可以根据需要灵活配置：

#### **仅启用自适应噪声版本**
```bash
python3 main.py --exp_id="dynamic_adaptive" --use_dynamic_model=True --adaptive_noise=True \
  --progressive_training=False --noise_adapt_rate=0.05 --flip_samples=7 --miw=0.0000001 \
  --noise=0.1 --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc \
  --n_bits=100 --n_epochs=10
```

#### **仅启用渐进式训练版本**
```bash
python3 main.py --exp_id="dynamic_progressive" --use_dynamic_model=True --adaptive_noise=False \
  --progressive_training=True --noise_min=0.01 --noise_max=0.3 --flip_samples=7 --miw=0.0000001 \
  --noise=0.1 --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc \
  --n_bits=100 --n_epochs=10
```

DynamicIABF训练过程中，模型会通过观察验证损失和训练损失的变化，自动调整噪声水平，寻找最佳的训练路径。渐进式训练则会从低噪声环境开始，逐步增加噪声水平，帮助模型更有效地学习在高噪声环境中传输图像的能力。

### 基本示例：训练IABF模型

以下是训练BSC噪声为0.1的100位IABF模型(MNIST数据集)的完整流程：

```bash
# 步骤1：下载MNIST数据集
python3 data_setup/download.py mnist

# 步骤2：生成对应的tfrecords文件
python3 data_setup/convert_to_records.py --dataset=mnist

# 步骤3：训练IABF模型
python3 main.py --exp_id="iabf_test" --flip_samples=7 --miw=0.0000001 --noise=0.1 \
  --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 \
  --n_epochs=10 --batch_size=100 --lr=0.001 --optimizer=adam --is_binary=False \
  --dec_arch="500,500" --enc_arch="500,500" --reg_param=0.0001
```

### 二值化MNIST示例

```bash
# 下载二值化MNIST数据集
python3 data_setup/download.py BinaryMNIST

# 生成对应的tfrecords文件
python3 data_setup/convert_to_records.py --dataset=BinaryMNIST

# 训练模型（注意设置is_binary=True）
python3 main.py --exp_id="iabf_binary" --flip_samples=7 --miw=0.0000001 --noise=0.1 \
  --datadir=./data --datasource=BinaryMNIST --is_binary=True --channel_model=bsc \
  --n_bits=100 --n_epochs=10 --batch_size=100
```

### 使用其他数据集

CIFAR10示例：
```bash
python3 main.py --exp_id="iabf_cifar" --flip_samples=7 --miw=0.0000001 --noise=0.1 \
  --datadir=./data --datasource=cifar10 --channel_model=bsc --n_bits=500 --n_epochs=50 \
  --batch_size=100 --dec_arch="1000,1000" --enc_arch="1000,1000" --reg_param=0.0001
```

---

## 📊 模型比较

本项目实现了四种不同的神经网络模型用于图像通信：DynamicIABF、IABF、NECST和UAE。

### 1. 模型简介

#### DynamicIABF (Dynamic Infomax Adversarial Bit Flip)
DynamicIABF是IABF的增强版本，在原有架构基础上增加了两个创新性机制：
1. **自适应噪声水平调整**：通过监测训练和验证过程中的损失变化实时调整噪声水平。这一机制解决了传统固定噪声训练的局限性，使模型能够自动寻找最佳的训练难度，类似于人类教育中的"适应性学习"。
2. **渐进式训练策略**：从低噪声环境逐步适应高噪声环境。这一策略基于认知科学中的"渐进式学习"理论，即通过由简到难的学习过程获取更强的泛化能力。

这两种机制协同工作，使DynamicIABF能够在任意噪声水平下实现稳健的图像传输性能，尤其适合实际通信环境中噪声水平多变的情况。研究表明，这种动态调整能力显著提高了模型在极端噪声情况下的表现。

**创新原理**：DynamicIABF的工作原理源于对传统训练方法的两个关键观察：
- 固定噪声下训练的模型在未见噪声水平下表现不佳
- 从零开始直接适应高噪声环境非常困难

为解决这些问题，DynamicIABF采用了类似于"课程学习"的策略，但增加了自适应调整机制，使噪声水平根据模型当前状态动态变化，避免训练过程陷入局部最优。技术上，模型通过监控验证损失的变化趋势，在损失减小时适度增加噪声水平，而在损失增加时降低噪声水平，保持训练过程的"最佳挑战区"。

#### IABF (Infomax Adversarial Bit Flip)
IABF是一种结合了信息最大化原则和对抗性训练的通信方法。它通过以下方式工作：
1. 编码器将输入图像压缩为固定长度的比特序列
2. 这些比特通过噪声信道传输（如BSC或BEC）
3. 解码器尝试从噪声比特中恢复原始图像
4. 对抗性训练过程：通过学习哪些比特对于成功解码最为关键，对这些比特进行特殊保护。

IABF利用"信息瓶颈"理论，找到图像中最信息密集的元素，并优先保护这些信息。此外，它还通过互信息项的显式优化，确保编码保留了原始图像的最大信息量。

#### NECST (Neural Error Correction and Source-channel coding for Transmission)
NECST是一种将源编码和信道编码联合优化的神经网络方法。它也采用编码器-信道-解码器架构，但不同的是，它直接优化重建误差，而不像IABF那样加入信息理论约束和对抗训练。NECST结构相对简单，但在某些场景下表现出色。

#### UAE (Unconstrained Auto-Encoder)
UAE是最基本的神经自编码器模型，不做任何形式的约束。它没有针对不同噪声信道的特殊优化，因此通常作为基线模型，用于比较其他方法的性能提升效果。

### 2. 训练参数对比

四种模型的关键参数对比：

| 参数 | DynamicIABF | IABF | NECST | UAE | 说明 |
|------|-------------|------|-------|-----|------|
| use_dynamic_model | True | False | False | False | 是否使用动态模型 |
| adaptive_noise | True/False | False | False | False | 是否使用自适应噪声调整 |
| progressive_training | True/False | False | False | False | 是否使用渐进式训练 |
| flip_samples | > 0 (如7) | > 0 (如7) | 0 | 0 | 对抗性位翻转的数量 |
| miw | > 0 (如0.0000001) | > 0 (如0.0000001) | 0 | 0 | 互信息权重 |
| noise | 初始值如0.1 | > 0 (如0.1) | > 0 (如0.1) | 0 | 训练时的信道噪声 |
| noise_min | > 0 (如0.01) | - | - | - | 渐进式训练最小噪声 |
| noise_max | > 0 (如0.3) | - | - | - | 渐进式训练最大噪声 |
| test_noise | > 0 (如0.1) | > 0 (如0.1) | > 0 (如0.1) | 0 | 测试时的信道噪声 |

完整训练命令示例：

#### DynamicIABF 训练命令
```bash
# 同时启用自适应噪声和渐进式训练
python3 main.py --exp_id="dynamic_iabf_test" --use_dynamic_model=True --adaptive_noise=True --progressive_training=True --noise_min=0.01 --noise_max=0.3 --noise_adapt_rate=0.05 --flip_samples=7 --miw=0.0000001 --noise=0.1 --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100 --lr=0.001 --optimizer=adam --is_binary=False --dec_arch="500,500" --enc_arch="500,500" --reg_param=0.0001

# 仅启用自适应噪声
python3 main.py --exp_id="dynamic_iabf_adaptive" --use_dynamic_model=True --adaptive_noise=True --progressive_training=False --noise_adapt_rate=0.05 --flip_samples=7 --miw=0.0000001 --noise=0.1 --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100

# 仅启用渐进式训练
python3 main.py --exp_id="dynamic_iabf_progressive" --use_dynamic_model=True --adaptive_noise=False --progressive_training=True --noise_min=0.01 --noise_max=0.3 --flip_samples=7 --miw=0.0000001 --noise=0.1 --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100
```

#### IABF 训练命令
```bash
python3 main.py --exp_id="iabf_test10" --flip_samples=7 --miw=0.0000001 --noise=0.1 --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100 --lr=0.001 --optimizer=adam --is_binary=False --dec_arch="500,500" --enc_arch="500,500" --reg_param=0.0001
```

#### NECST 训练命令
```bash
python3 main.py --exp_id="necst_test10" --flip_samples=0 --miw=0.0 --noise=0.1 --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100 --lr=0.001 --optimizer=adam --is_binary=False --dec_arch="500,500" --enc_arch="500,500" --reg_param=0.0001
```

#### UAE 训练命令
```bash
python3 main.py --exp_id="uae_test10" --flip_samples=0 --miw=0.0 --noise=0.0 --test_noise=0.0 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100 --lr=0.001 --optimizer=adam --is_binary=False --dec_arch="500,500" --enc_arch="500,500" --reg_param=0.0001
```

### 3. 训练方法

#### DynamicIABF 训练流程
1. **初始化阶段**：根据设置确定起始噪声水平（渐进式训练使用`noise_min`，自适应噪声使用`noise`）
2. **编码阶段**：训练编码器将输入图像编码为二进制向量
3. **对抗翻转**：对编码向量应用对抗性位翻转（最具信息量的位）
4. **噪声模拟**：通过自适应或渐进调整的噪声水平模拟信道传输
5. **解码阶段**：训练解码器恢复原始图像
6. **互信息优化**：使用互信息损失项指导编码器学习更鲁棒的表示
7. **噪声动态调整**：
   - **批间调整**：每隔固定批次，基于训练损失变化调整噪声水平
   - **轮次间调整**：每个训练轮次结束后，基于验证损失变化进一步优化噪声水平
   - **渐进式调整**：如果启用渐进式训练，随着训练进度线性增加噪声水平
8. **最佳模型选择**：根据验证集性能选择最佳检查点

DynamicIABF的训练过程可以可视化为一个动态适应的过程。相比传统方法，它不是在固定的噪声环境中训练模型，而是创建了一个"自适应噪声梯度"，使模型首先建立基本的通信能力，然后逐渐增强对噪声的适应能力，最终实现在广泛噪声范围内的优异性能。这类似于运动员先掌握基本技能，再逐步增加训练强度的过程。

**技术实现**：DynamicIABF的自适应系统通过以下机制实现：
```python
# 自适应噪声调整（简化示例）
if current_loss < previous_loss:
    noise_level = min(noise_level * (1 + adapt_rate), max_noise)  # 增加难度
else:
    noise_level = max(noise_level * (1 - adapt_rate), min_noise)  # 降低难度

# 渐进式训练调整（简化示例）
progress = current_epoch / total_epochs
noise_level = min_noise + (max_noise - min_noise) * progress  # 线性增加难度
```

这种动态训练机制使DynamicIABF在各种噪声条件下都能保持出色的性能，特别是在未知噪声水平的实际应用场景中。

#### IABF 训练流程
1. 训练编码器将输入图像编码为二进制向量
2. 对编码向量应用对抗性位翻转（最具信息量的位）
3. 通过模拟噪声信道传输
4. 训练解码器恢复原始图像
5. 使用互信息损失项指导编码器学习更鲁棒的表示

#### NECST 训练流程
1. 训练编码器将输入图像编码为二进制向量
2. 通过添加随机噪声而非对抗性噪声来模拟信道条件
3. 训练解码器恢复原始图像
4. 不使用互信息约束，仅使用重构损失

#### UAE 训练流程
1. 训练传统的自编码器结构
2. 不添加任何噪声
3. 不使用互信息约束
4. 仅优化重构损失

### 4. 关键区别

- **噪声水平调整**：
  - **DynamicIABF**：实现了两种突破性的噪声调整机制：
    1. **基于损失的自适应调整**：通过监控训练和验证损失变化动态调整噪声水平。当模型适应当前噪声水平（损失减小）时，系统增加噪声水平以进一步挑战模型；当模型处于困难阶段（损失增加）时，系统降低噪声水平以确保稳定学习。这种方法类似于"自适应难度系统"，确保训练始终在最有效的难度级别进行。
    2. **基于进度的渐进式调整**：从较低的噪声水平开始训练，随着训练进度按照预设的曲线（默认为线性）逐渐增加噪声水平。这种方法模拟了"课程学习"，使模型首先掌握基本的编解码能力，然后再面对更具挑战性的噪声环境。
    
    两种机制可以单独使用，也可以组合使用以获得更好的效果。实验表明，组合使用时，自适应机制可以在渐进式基准上进行微调，进一步优化训练过程。
  - **IABF**：固定的噪声水平，不随训练过程调整
  - **NECST**：固定的噪声水平，不随训练过程调整
  - **UAE**：无噪声训练
  
- **互信息最大化**：
  - **DynamicIABF/IABF**：通过互信息损失项明确地最大化输入和编码之间的互信息，但DynamicIABF通过动态噪声调整能够在更广泛的噪声条件下保持高互信息
  - **NECST/UAE**：没有这一约束
  
- **对抗训练**：
  - **DynamicIABF/IABF**：使用对抗性位翻转来提高鲁棒性，DynamicIABF在此基础上增加了适应性训练
  - **NECST**：使用随机噪声
  - **UAE**：无噪声
  
- **噪声类型**：
  - **DynamicIABF**：对抗性位翻转 + 自适应/渐进式动态调整的随机信道噪声
  - **IABF**：对抗性位翻转 + 固定随机信道噪声
  - **NECST**：仅固定随机信道噪声
  - **UAE**：无噪声
  
- **优化目标**：
  - **DynamicIABF/IABF**：重构损失 + 互信息最大化，但DynamicIABF增加了对噪声水平适应性的隐式优化
  - **NECST/UAE**：仅重构损失
  
- **泛化能力**：
  - **DynamicIABF**：卓越的泛化能力，能够适应未见过的噪声水平
  - **IABF**：对接近训练噪声水平的情况有良好性能
  - **NECST**：对接近训练噪声水平的情况有一定性能
  - **UAE**：在有噪声环境中性能迅速下降
  
- **训练复杂度**：
  - **DynamicIABF**：较高（需要额外的噪声调整机制和噪声水平跟踪）
  - **IABF**：中等
  - **NECST**：中等
  - **UAE**：较低
  
- **工程实现难度**：
  - **DynamicIABF**：需要在训练过程中实时监控模型性能并调整噪声参数
  - **IABF/NECST/UAE**：标准训练流程，参数固定

### 5. 技术细节对比

#### DynamicIABF模型
- **核心创新机制**：DynamicIABF引入了两个互补的噪声调整机制：
  1. **自适应噪声调整算法**：实现为类似PID（比例-积分-微分）控制器的机制，通过计算损失函数变化率实时调整噪声水平。算法会根据损失下降的速度和幅度，按比例调整噪声水平，在保证训练稳定性的前提下最大化挑战难度。
  
  2. **渐进式噪声计划**：使用预设的噪声增长函数（默认为线性，但也支持其他曲线如指数或S形），根据当前训练进度计算合适的噪声水平。该机制尤其适合训练初期，使模型能够首先掌握基本的编解码能力。

- **动态参数追踪**：系统在训练过程中持续追踪和记录以下指标：
  - 当前噪声水平
  - 训练损失趋势
  - 验证损失趋势
  - 噪声调整历史

- **噪声波动控制**：为防止噪声水平剧烈波动，系统引入了两种稳定机制：
  - **调整频率限制**：批次级别的调整仅在特定间隔（默认每10个批次）执行一次
  - **调整幅度限制**：每次调整的噪声变化幅度受`noise_adapt_rate`参数控制

- **训练优化器适配**：DynamicIABF模型保持与IABF相同的编码器和解码器架构，但对优化过程进行了增强，特别是在以下方面：
  - 批量归一化层的统计量更新策略改进，适应噪声水平变化
  - 学习率调度策略优化，考虑噪声水平变化带来的梯度波动

- **数据结构与实现**：
  ```python
  class DynamicIABF(IABF):
      def __init__(self, ...):
          # 继承IABF的基础结构
          super().__init__(...)
          
          # 动态模型特有属性
          self.use_dynamic_model = FLAGS.use_dynamic_model
          self.adaptive_noise = FLAGS.adaptive_noise
          self.progressive_training = FLAGS.progressive_training
          self.noise_min = FLAGS.noise_min
          self.noise_max = FLAGS.noise_max
          self.noise_adapt_rate = FLAGS.noise_adapt_rate
          self.current_noise = FLAGS.noise  # 初始噪声水平
          self.previous_loss = float('inf')  # 用于自适应调整
          self.current_epoch = 0  # 当前训练轮次
          
      def adaptive_noise_level(self, current_loss):
          """基于训练损失变化自适应调整噪声水平"""
          # 实现算法详见代码
          
      def progressive_training_noise(self):
          """基于训练进度计算渐进式噪声水平"""
          # 实现算法详见代码
  ```

#### IABF模型
- **对抗位翻转机制**：IABF会计算每个位对重构质量的梯度，并有针对性地翻转对重构影响最大的若干位。这种方法不是随机的，而是有选择性地攻击编码的薄弱环节。
- **翻转策略**：通过计算重构损失对每个编码位的梯度，确定哪些位对重构质量最重要，然后翻转这些位，相当于一种"最坏情况"的噪声模拟。
- **互信息估计**：使用基于VIMCO（Variational Inference for Monte Carlo Objectives）算法的变分下界来估计输入和编码之间的互信息。
- **编码优化**：通过同时优化重构质量和互信息，IABF能够学习更加鲁棒的编码策略，其中重要信息分布在多个编码位上，而不是集中在少数几个位。

#### NECST模型
- **随机噪声机制**：NECST采用二进制对称信道(BSC)或二进制擦除信道(BEC)中的随机噪声模型，以固定概率随机翻转或擦除编码位。
- **噪声适应**：通过在训练过程中暴露于随机信道噪声，模型学习适应噪声环境，但没有明确针对"最坏情况"的优化。
- **没有互信息约束**：NECST不显式优化互信息，因此编码可能不够紧致或均匀分布。

#### UAE模型
- **无噪声训练**：UAE完全在无噪声条件下训练，代表理想的通信环境。
- **标准自编码器架构**：使用与IABF和NECST相同的编码器和解码器架构，但没有任何噪声模拟或互信息约束。
- **性能基准**：UAE提供了无噪声条件下的上限性能，作为其他两种模型的比较基准。

### 6. 损失函数对比

#### DynamicIABF损失函数组成
```
总损失 = 重构损失 + miw * 互信息损失 + wadv * 对抗重构损失 + 噪声适应性隐式调整
```
- **重构损失**：原始输入与解码输出之间的误差（L2距离或交叉熵）
- **互信息损失**：通过VIMCO估计的输入与编码之间互信息的负值
- **对抗重构损失**：翻转最关键位后的解码输出与原始解码输出之间的误差
- **噪声适应性隐式调整**：虽然不是显式的损失项，但通过动态调整噪声水平，系统隐式地优化了模型在多种噪声条件下的性能

DynamicIABF的损失函数与IABF基本相同，但关键区别在于训练过程中噪声水平的动态调整。这种调整实质上创建了一个"元优化"层，使模型不仅优化于特定噪声水平，而是在噪声水平分布上进行优化。从信息论角度，这相当于增加了一个额外约束：模型必须在噪声分布而非单点噪声值上最大化互信息。

**技术实现**：噪声动态调整通过以下两个方法实现：
1. **自适应调整函数**：
   ```python
   def adaptive_noise_level(self, current_loss):
       if current_loss < self.previous_loss:
           # 损失减小，增加噪声水平挑战模型
           new_noise = min(self.current_noise * (1.0 + self.noise_adapt_rate), self.noise_max)
       else:
           # 损失增加，减少噪声水平稳定训练
           new_noise = max(self.current_noise * (1.0 - self.noise_adapt_rate), self.noise_min)
       self.previous_loss = current_loss
       return new_noise
   ```

2. **渐进式调整函数**：
   ```python
   def progressive_training_noise(self):
       # 根据训练进度线性增加噪声水平
       progress = min(1.0, self.current_epoch / float(FLAGS.n_epochs))
       noise_level = self.noise_min + (self.noise_max - self.noise_min) * progress
       return noise_level
   ```

这两种机制创建了一种强大的噪声适应性训练范式，使模型能够在广泛的噪声条件下表现良好。

#### IABF损失函数组成
```
总损失 = 重构损失 + miw * 互信息损失 + wadv * 对抗重构损失
```

#### NECST损失函数组成
```
总损失 = 重构损失
```
- **重构损失**：原始输入与通过随机噪声信道后解码输出之间的误差

#### UAE损失函数组成
```
总损失 = 重构损失
```
- **重构损失**：原始输入与解码输出之间的误差，无任何额外项

### 7. 适用场景

#### DynamicIABF优势场景
- **动态噪声环境**：通信信道噪声水平频繁变化的场景，如移动通信、无线传感器网络
- **多样化噪声条件**：需要同时适应多种不同噪声水平的场合，如多用户、多场景应用
- **未知噪声环境**：事先不确定实际部署环境噪声特性的情况
- **高价值信息传输**：对信息完整性有极高要求的场景，如医疗图像、金融数据传输
- **自适应通信系统**：需要根据信道状况动态调整传输策略的系统
- **渐进式部署策略**：希望从低噪声环境开始，逐步扩展到高噪声环境的应用

DynamicIABF尤其适合通信噪声条件不稳定或难以预测的场景。由于其自适应能力，它能够在各种噪声水平下保持良好性能，而不需要针对每种噪声条件单独训练模型。这使得DynamicIABF成为构建通用鲁棒通信系统的理想选择。

#### IABF优势场景
- **高噪声通信环境**：如无线通信、深空通信等高噪声环境
- **需要强鲁棒性的场景**：医疗图像传输、安全监控系统等对重建质量要求高的场景
- **资源受限的通信**：卫星通信、IoT设备通信等带宽和功率受限的场景
- **抗恶意干扰通信**：可能面临恶意干扰的通信环境

#### NECST优势场景
- **中等噪声环境**：有一定随机噪声但不太严重的通信环境
- **计算资源有限的设备**：由于不需要计算互信息和对抗性位翻转，计算负担较轻
- **对编码效率要求不高的场景**：有足够带宽但仍需一定噪声鲁棒性的场合

#### UAE优势场景
- **低噪声或无噪声环境**：如有线局域网、光纤通信等高质量通信信道
- **对延迟敏感的实时通信**：计算复杂度最低，适合实时性要求高的场景
- **训练资源有限的场景**：训练过程最简单，不需要复杂的损失函数优化

## 模型输出

模型训练结束后会产生以下输出和返回值：

| 输出变量/文件 | 描述 | 存储位置 |
|--------------|------|---------|
| model.ckpt-{global_step} | 训练好的模型参数 | `FLAGS.logdir/model.ckpt-{global_step}` |
| best_ckpt | 验证集损失最低的检查点 | 训练函数返回值 |
| epoch_train_losses | 每个epoch的训练损失 | 训练函数返回值 |
| epoch_valid_losses | 每个epoch的验证损失 | 训练函数返回值 |
| test_loss | 测试集上的L2损失 | 测试函数输出及返回值 |
| mi = (-hy_-hy_x_) / num_batches | 互信息估计 | 测试函数输出 |
| distribution.pdf | 边缘概率分布直方图 | `FLAGS.outdir/distribution.pdf` |
| reconstructions.png | 重构图像对比 | `FLAGS.outdir/reconstructions.png` |
| reconstr.pkl | 保存的重构数据 | `FLAGS.outdir/reconstr.pkl` |
| markov_chain_samples.png | 马尔可夫链生成样本 | `FLAGS.outdir/markov_chain_samples.png` |
| config.json | 配置参数记录 | `FLAGS.outdir/config.json` |
| log.txt | 训练日志 | `FLAGS.outdir/log.txt` |
| TensorBoard事件文件 | 训练可视化数据 | `FLAGS.outdir/train/` 和 `valid/` |

## 性能比较

为了比较IABF、NECST和UAE三种模型的性能，我们提供了可视化工具`plot.py`。

### 比较流程

1. 使用统一参数训练三种模型（保持n_bits、n_epochs等参数一致）
2. 为每种模型指定不同的`exp_id`
3. 使用`plot.py`生成比较图表：

```bash
python3 plot.py --model_dirs "mnist/miw_1e-07_flip_7_bits_100_epochs_10/iabf_test10" "mnist/miw_0.0_flip_0_bits_100_epochs_10/necst_test10" "mnist/miw_0.0_flip_0_bits_100_epochs_10/uae_test10" --model_names "IABF" "NECST" "UAE" --plot_type all
```

### 评估指标

1. **重构质量**：通过MSE和PSNR评估重构图像质量
2. **互信息**：比较编码器输入和输出之间的互信息
3. **抗噪性**：在不同噪声水平下的重构质量表现
4. **边缘分布**：编码器输出的概率分布特性
5. **收敛速度**：模型收敛所需的训练轮数

### 预期结果

- **IABF**：在有噪声条件下具有最佳重构质量和最高互信息
- **NECST**：在有噪声条件下表现优于UAE但不如IABF
- **UAE**：无噪声条件下表现良好，有噪声条件下性能显著下降

## 可视化工具

### 模块化可视化系统

本项目提供了一个模块化、可扩展的可视化系统，用于分析和比较DynamicIABF、IABF、NECST和UAE等模型的性能。该系统包含以下主要组件：

1. **主入口点**：`plot.py` 作为兼容性入口，内部调用模块化的 `visual` 包实现功能
2. **核心可视化类**：`ModelVisualizer` 负责协调所有可视化功能
3. **专用可视化模块**：针对学习曲线、分布比较、互信息分析、重建质量等的特定可视化器
4. **支持工具**：数据提取、模型目录查找和样式设置等工具函数

### 功能概述

可视化系统支持以下功能：

- **学习曲线分析**：训练和验证损失随训练过程的变化
- **互信息分析**：模型间互信息比较及其随训练的变化趋势
- **分布可视化**：编码的边缘概率分布比较
- **重建质量评估**：原始图像与重建图像对比及PSNR/SSIM指标计算
- **测试指标对比**：多模型测试性能和噪声鲁棒性分析
- **DynamicIABF特性分析**：自适应噪声和渐进式训练的效果可视化

### 使用方法

#### 命令行使用

```bash
# 自动发现并可视化所有模型
python plot.py --auto_discover --output_dir ./plots/

# 比较特定模型
python plot.py --model_dirs "DynamicIABF/mnist/exp1" "IABF/mnist/exp2" "NECST/mnist/exp3" --model_names "DynamicIABF" "IABF" "NECST" --plot_type all

# 仅生成特定类型图表
python plot.py --auto_discover --plot_type learning --output_dir ./plots/

# 指定数据集
python plot.py --dataset mnist --auto_discover --output_dir ./plots/
```

#### Python API使用

```python
from visual.core import ModelVisualizer

# 创建可视化器
visualizer = ModelVisualizer(
    base_dir="./results/", 
    model_dirs=["DynamicIABF/mnist/exp1", "IABF/mnist/exp2"],
    model_names=["DynamicIABF", "IABF"],
    dataset_name="mnist"
)

# 生成特定图表
visualizer.plot_learning_curves("./plots/learning.png")
visualizer.plot_mutual_information("./plots/mi.png")
visualizer.plot_reconstruction_samples("./plots/reconstruction.png")

# 生成所有支持的图表
visualizer.plot_all("./plots/")
```

### 支持的图表类型

可视化系统支持以下图表类型：

1. **学习曲线** (`learning`)：训练和验证损失曲线
2. **互信息比较** (`mutual_information`)：各模型互信息值对比
3. **互信息随时间变化** (`mi_over_time`)：互信息随训练轮次的变化
4. **分布比较** (`distribution`)：激活值分布直方图
5. **重建样本比较** (`reconstruction`)：原始与重建图像对比
6. **重建误差表格** (`errors_table`)：重建质量指标可视化表格
7. **测试指标比较** (`metrics`)：测试损失、准确率等指标比较
8. **噪声分析** (`noise_analysis`)：不同噪声水平下的性能曲线
9. **DynamicIABF特性分析** (`dynamic_iabf`)：DynamicIABF模型特有特性分析

### 命令行参数

```
--base_dir: 结果文件基础目录（默认: "./results/"）
--model_dirs: 模型结果子目录列表
--model_names: 图例中使用的模型名称
--model_types: 要显示的模型类型（DynamicIABF/IABF/NECST/UAE）
--output_dir: 保存图像的输出目录（默认: "./plots/"）
--auto_discover: 自动发现模型目录
--plot_type: 图表类型 [all, learning, distribution, mutual_information, mi_over_time, reconstruction, errors_table, metrics, noise_analysis, dynamic_iabf]
--extract_data: 强制提取和更新测试数据
--num_samples: 要提取的测试样本数量（默认: 10）
--dataset: 数据集类型 [mnist, binary_mnist, cifar10, svhn, omniglot, celeba]
```

### 使用示例

#### 自动发现并可视化所有模型（包括DynamicIABF）

```bash
python plot.py --auto_discover --dataset mnist --output_dir "./plots/"
```

#### 对比不同模型配置性能

```bash
python plot.py --model_dirs "DynamicIABF/mnist/dynamic_iabf_test" "IABF/mnist/iabf_test" "NECST/mnist/necst_test" --model_names "DynamicIABF" "IABF" "NECST" --plot_type all --output_dir "./plots_comparison/"
```

#### 仅比较DynamicIABF的不同配置

```bash
python plot.py --model_dirs "DynamicIABF/mnist/adaptive_only" "DynamicIABF/mnist/progressive_only" "DynamicIABF/mnist/combined" --model_names "仅自适应" "仅渐进式" "组合模式" --plot_type all --output_dir "./plots_dynamic/"
```

#### 专门分析噪声鲁棒性

```bash
python plot.py --auto_discover --model_types "DynamicIABF" "IABF" --plot_type noise_analysis --output_dir "./plots_noise/"
```

### 输出图表

系统会在指定的输出目录下生成以下图表：

- `learning_curves.png`：学习曲线对比
- `mutual_information.png`：互信息比较
- `mutual_information_over_time.png`：互信息随时间变化趋势
- `distribution_comparison.png`：编码分布直方图
- `reconstruction_samples.png`：重建样本质量对比
- `reconstruction_errors_table.png`：重建误差指标表格
- `test_metrics.png`：测试指标条形图
- `noise_analysis.png`：噪声鲁棒性曲线
- `dynamic_iabf_features.png`：DynamicIABF特性分析（仅适用于DynamicIABF模型）

### 技术特点

1. **模块化架构**：每个功能封装在专用模块中，便于扩展和维护
2. **兼容性设计**：保留与原始代码的兼容性，确保现有流程不受影响
3. **统一数据管理**：统一的测试数据提取和管理机制
4. **高质量输出**：支持出版级图表，包括矢量格式输出
5. **丰富的自定义选项**：提供多种参数配置可视化效果

## 更新日志

### 20250429更新：优化可视化组件和指标生成

1. **metrics.pkl文件自动生成功能**
   - 创建visual/generate_metrics.py脚本用于从训练日志和重建结果中提取指标
   - 在necst.py中集成自动生成metrics.pkl功能，在test阶段自动调用
   - 支持MSE, MAE, PSNR, SSIM等重建质量指标自动计算
   - 保存互信息和分布数据以便可视化使用

2. **解决可视化图表问题**
   - 修复distribution_comparison.pdf无法正确显示的问题
   - 增强visual/plots/distribution.py，支持从多种来源读取分布数据
   - 当真实分布数据不可用时，根据模型类型生成合理的模拟分布
   - 自动保存分布数据为.npy格式方便下次使用

3. **可视化流程优化**
   - 简化模型比较可视化流程，改进数据读取机制
   - 增强可视化组件对数据缺失的容错能力
   - 保证四种模型的可视化图表能够完整生成

这些更新大大提高了模型评估和可视化的便捷性，使得模型性能比较更加直观和全面。用户可以使用visual/generate_metrics.py脚本为已训练模型批量生成指标文件，或者直接通过测试阶段自动生成，然后通过可视化工具进行全面分析。

### 2025年5月25日更新：自动化对照训练脚本

1. **新增自动化对照训练脚本**
   - 添加`run_comparison.sh`脚本实现完整训练流程的一键式执行
   - 支持数据准备、多模型训练和结果可视化的自动化流程
   - 使用单一实验ID管理所有相关文件和目录

2. **实验管理优化**
   - 结构化日志记录系统，跟踪每个训练阶段
   - 按实验ID组织输出结果，方便对比分析
   - 自动利用plot.py的增强功能进行结果可视化

3. **使用便利性提升**
   - 添加参数化配置，支持自定义数据集、编码比特数和训练轮数
   - 提供易于理解的使用示例和结果展示方法
   - 预留实验框架以支持更复杂的对比测试

4. **文档更新**
   - 在README中详细介绍脚本功能和使用方法
   - 添加多种使用场景的示例命令
   - 说明结果目录结构和日志文件组织

该脚本与之前的plot.py增强功能协同工作，共同提供了从训练到可视化的一站式实验解决方案，大大简化了多模型对比实验的流程。

### 2025年5月20日更新：plot.py可视化工具增强

1. **增强的模型目录查找功能**
   - `find_model_dirs`函数完全支持新旧目录结构
   - 能自动识别不同目录结构下的模型类型
   - 根据config.json内容智能识别模型为DynamicIABF/IABF/NECST/UAE

2. **自动发现模型目录功能**
   - 新增`--auto_discover`命令行参数，自动查找符合条件的模型
   - 支持通过`--model_types`参数过滤特定类型的模型
   - 简化了可视化不同模型的流程，无需手动指定每个模型路径

3. **智能模型名称生成**
   - 根据目录结构自动生成便于识别的模型名称
   - 格式为`{模型类型}-{数据集}-{实验ID}`
   - 从实验目录名中自动提取关键信息，忽略时间戳

4. **文档更新**
   - 更新README中的完整训练流程教程
   - 添加新的可视化命令示例
   - 说明如何利用新功能简化模型对比分析

这些增强功能使得模型结果的可视化和对比更加便捷，特别是在管理多个不同类型模型的实验时，大大减少了手动指定路径的工作量。

### 2025年5月15日更新：DynamicIABF模型引入

1. **DynamicIABF模型正式发布**
   - 引入自适应噪声水平调整和渐进式训练策略
   - 显著提高了模型在多种噪声环境下的鲁棒性
   - 新增命令行参数控制自适应和渐进式功能

2. **核心创新功能**
   - `adaptive_noise_level()` 方法实现基于损失反馈的噪声动态调整
   - `progressive_training_noise()` 方法实现基于训练进度的噪声增长规划
   - 优化的训练循环支持实时噪声参数更新

3. **代码架构优化**
   - 保持与原IABF模型的兼容性
   - 模块化设计使新功能可选择性启用
   - 增强的监控和日志功能，跟踪噪声调整过程

4. **文档更新**
   - 更新README添加DynamicIABF详细说明
   - 提供不同配置组合的使用示例
   - 添加技术原理和应用场景分析

### 2025年4月22日更新：可视化与数据管理优化

1. **测试数据统一管理**
   - 将测试数据统一存储在`data`目录的`mnist_test_data.pkl`文件中
   - 避免重复存储测试数据，节省存储空间
   - 确保所有模型使用相同测试样本

2. **`plot.py`功能增强**
   - 集成测试数据提取功能
   - 自动确保测试数据可用
   - 改进重构图像显示效果

3. **命令行参数增强**
   ```bash
   --extract_data (BOOL): 强制重新提取测试数据
   --num_samples (INT): 提取的测试样本数量（默认10）
   ```

4. **自动检测和提取**
   - 自动检测测试数据文件是否存在
   - 可选择强制更新已存在的测试数据文件

### 2025年4月23日更新：可视化工具增强

- 增加对多数据集的可视化支持（MNIST、BinaryMNIST、CIFAR10、SVHN等）
- 自动适配不同数据集的图像尺寸和颜色格式
- 优化不同类型图像的显示效果
- 改进模型标签显示位置
- 提供简化的命令行接口：
  ```bash
  # 使用CIFAR10数据集进行可视化
  python3 plot.py --dataset cifar10 --model_dirs "cifar10/model1" "cifar10/model2" --model_names "IABF" "NECST"
  
  # 自动发现和可视化所有模型结果
  python3 plot.py --dataset svhn --auto_discover --model_types "IABF" "NECST" "UAE"
  ```
- 支持为每个数据集提取和保存测试样本
- 增强PSNR计算精度

### 20250426：优化results输出目录结构

为了更好地组织实验结果文件，对results输出目录结构进行了如下调整：

- **三级目录结构**：按照`{模型类型}/{数据集}/{日期时间戳}_{参数简写}_{实验ID}/`进行组织
- **模型类型分组**：将模型结果按DynamicIABF、IABF、NECST和UAE四种类型进行分类存储
- **参数简写**：采用简化的参数命名约定，如`b100`(100位编码)、`m1e-7`(miw值)、`f7`(7位翻转)等
- **时间戳标记**：使用`YYYYMMDD`格式的时间戳标记每个实验的执行日期

示例目录结构：
```
results/
├── IABF/
│   └── mnist/
│       └── 20250426_b100_m1e-7_f7_iabf_test10/
├── NECST/
│   └── mnist/
│       └── 20250426_b100_n0.1_necst_test10/
└── DynamicIABF/
    └── mnist/
        └── 20250426_b100_m1e-7_f7_ap_dynamic_full/
```

同时，对plot.py也进行了功能增强，支持新的目录结构：

- **增强的模型发现功能**：`find_model_dirs`函数支持新旧目录结构，能自动查找并识别不同类型的模型
- **自动识别模型类型**：通过config.json内容识别模型类型（DynamicIABF/IABF/NECST/UAE）
- **友好的命令行接口**：新增`--auto_discover`参数自动发现模型目录，简化使用流程
- **模型类型过滤**：通过`--model_types`参数过滤特定类型的模型，如：
  ```bash
  python3 plot.py --auto_discover --model_types "DynamicIABF" "IABF"
  ```
- **自动生成便于识别的模型名称**：格式为`{ModelType}-{dataset}-{experiment_id}`

这些更新使得实验结果组织更加条理清晰，也让不同模型的对比分析更加便捷。

详细说明请参考项目中的`directory_structure.md`文件。

---

## 完整训练流程教程

本教程详细介绍使用IABF框架完成训练的完整流程，包括数据准备、模型训练和结果可视化。

### 1. 环境和数据准备

```bash
# 检查数据目录
ls -la data/
ls -la data/mnist/

# 如果数据不存在，下载并转换MNIST数据集
python3 data_setup/download.py mnist
python3 data_setup/convert_to_records.py --dataset=mnist

# 创建结果目录 - 注意新的目录结构按模型类型组织
mkdir -p ./results/DynamicIABF/mnist ./results/IABF/mnist ./results/NECST/mnist ./results/UAE/mnist
```

### 2. 模型训练

#### DynamicIABF模型

```bash
# 训练完整版DynamicIABF（同时启用自适应噪声和渐进式训练）
python3 main.py --exp_id="dynamic_iabf_full" --use_dynamic_model=True --adaptive_noise=True --progressive_training=True --noise_min=0.01 --noise_max=0.3 --noise_adapt_rate=0.05 --flip_samples=7 --miw=0.0000001 --noise=0.1 --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100 --lr=0.001 --optimizer=adam --is_binary=False --dec_arch="500,500" --enc_arch="500,500" --reg_param=0.0001

# 训练仅启用自适应噪声的DynamicIABF
python3 main.py --exp_id="dynamic_iabf_adaptive" --use_dynamic_model=True --adaptive_noise=True --progressive_training=False --noise_adapt_rate=0.05 --flip_samples=7 --miw=0.0000001 --noise=0.1 --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100 --lr=0.001 --optimizer=adam --is_binary=False --dec_arch="500,500" --enc_arch="500,500" --reg_param=0.0001

# 训练仅启用渐进式训练的DynamicIABF
python3 main.py --exp_id="dynamic_iabf_progressive" --use_dynamic_model=True --adaptive_noise=False --progressive_training=True --noise_min=0.01 --noise_max=0.3 --flip_samples=7 --miw=0.0000001 --noise=0.1 --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100 --lr=0.001 --optimizer=adam --is_binary=False --dec_arch="500,500" --enc_arch="500,500" --reg_param=0.0001
```

#### IABF模型

```bash
python3 main.py --exp_id="iabf_test10" --flip_samples=7 --miw=0.0000001 --noise=0.1 --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100 --lr=0.001 --optimizer=adam --is_binary=False --dec_arch="500,500" --enc_arch="500,500" --reg_param=0.0001
```

#### NECST模型

```bash
python3 main.py --exp_id="necst_test10" --flip_samples=0 --miw=0.0 --noise=0.1 --test_noise=0.1 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100 --lr=0.001 --optimizer=adam --is_binary=False --dec_arch="500,500" --enc_arch="500,500" --reg_param=0.0001
```

#### UAE模型

```bash
python3 main.py --exp_id="uae_test10" --flip_samples=0 --miw=0.0 --noise=0.0 --test_noise=0.0 --datadir=./data --datasource=mnist --channel_model=bsc --n_bits=100 --n_epochs=10 --batch_size=100 --lr=0.001 --optimizer=adam --is_binary=False --dec_arch="500,500" --enc_arch="500,500" --reg_param=0.0001
```

### 3. 监控训练进度

```bash
# 查看训练日志 - 使用新的目录结构
tail -n 20 ./results/DynamicIABF/mnist/*/dynamic_iabf_full/log.txt
tail -n 20 ./results/IABF/mnist/*/iabf_test10/log.txt
tail -n 20 ./results/NECST/mnist/*/necst_test10/log.txt
tail -n 20 ./results/UAE/mnist/*/uae_test10/log.txt
```

### 4. 结果可视化

```bash
# 创建plots目录
mkdir -p ./plots

# 方法1：使用自动发现功能 - 自动查找所有模型类型
python3 plot.py --dataset mnist --auto_discover --output_dir "./plots/" --plot_type all

# 方法2：仅查找特定模型类型
python3 plot.py --dataset mnist --auto_discover --model_types "DynamicIABF" "IABF" --output_dir "./plots_compare/" --plot_type all

# 方法3：手动指定模型目录（使用相对于结果目录的路径）
python3 plot.py --dataset mnist --model_dirs "DynamicIABF/mnist/20250515_b100_ap_dynamic_iabf_full" "IABF/mnist/20250515_b100_m1e-7_f7_iabf_test10" --model_names "DynamicIABF" "IABF" --output_dir "./plots_manual/" --plot_type all

# 创建重建图像样本，并强制更新测试数据
python3 plot.py --dataset mnist --auto_discover --extract_data --num_samples 5 --output_dir "./plots_samples/" --plot_type reconstruction

# 生成噪声分析图表
python3 plot.py --dataset mnist --auto_discover --model_types "DynamicIABF" "IABF" "NECST" --output_dir "./plots_noise/" --plot_type noise
```



## 自动化对照训练脚本

为了简化多模型对照训练和对比分析的过程，项目提供了自动化脚本`run_comparison.sh`。此脚本可以一键完成上述完整训练流程，包括数据准备、四种模型训练和结果可视化，大大提高了实验效率。

### 脚本功能

`run_comparison.sh`具有以下主要功能：

1. **统一实验管理**：
   - 使用单一实验ID管理所有相关文件和目录
   - 自动创建结构化的日志和结果目录
   - 详细记录实验参数和执行过程

2. **完整流程自动化**：
   - 自动检查并下载所需数据集
   - 并行训练四种模型（DynamicIABF、IABF、NECST、UAE）
   - 自动使用plot.py生成全面的对比图表

3. **灵活参数配置**：
   - 支持配置数据集、编码比特数和训练轮数
   - 使用合理的默认参数简化操作
   - 保留扩展性以支持更复杂的实验设计

4. **实验结果分析**：
   - 生成多种可视化图表用于模型性能对比
   - 支持不同噪声水平下的性能测试
   - 提供实验结果的网页展示方法

### 使用方法

```bash
./run_comparison.sh <experiment_id> [dataset=mnist] [n_bits=100] [n_epochs=200] [learning_rate=0.001]
```

**参数说明**：
- `experiment_id`：**必填**参数，为本次对照试验的唯一标识符，将用于命名日志和结果目录
- `dataset`：可选参数，指定使用的数据集，默认为"mnist"
- `n_bits`：可选参数，指定编码比特数，默认为100
- `n_epochs`：可选参数，指定训练轮数，默认为200
- `learning_rate`：可选参数，指定学习率，默认为0.001

### 使用示例

基本用法：
```bash
# 使用默认参数运行对照实验，仅指定实验ID为"baseline_test"
./run_comparison.sh baseline_test
```

使用不同数据集：
```bash
# 使用CIFAR10数据集运行对照实验
./run_comparison.sh cifar_comparison cifar10
```

自定义比特数和训练轮数：
```bash
# 使用200个编码比特，训练15轮
./run_comparison.sh high_capacity mnist 200 15
```

进行全面实验：
```bash
# 在Omniglot数据集上进行长时间训练
./run_comparison.sh omniglot_full omniglot 300 50
```

自定义学习率：
```bash
# 使用更低的学习率0.0005进行训练
./run_comparison.sh low_lr_test mnist 100 200 0.0005
```

### 输出结果

脚本执行后，您将得到以下结果：

1. **日志文件**：位于`./logs/{实验ID}/`目录
   - `experiment.log`：总体实验日志
   - `data_preparation.log`：数据准备日志
   - `{model_type}_training.log`：各模型训练日志
   - `visualization.log`：可视化过程日志

2. **训练结果**：按照新的三级目录结构存储在`./results/`目录
   - `./results/DynamicIABF/{数据集}/{时间戳}_{参数}_{实验ID}_dynamic/`
   - `./results/IABF/{数据集}/{时间戳}_{参数}_{实验ID}_iabf/`
   - 以此类推...

3. **可视化图表**：位于`./plots/{实验ID}/`目录
   - 基本图表：学习曲线、互信息、分布对比等
   - 专门图表：噪声分析、重建样本对比
   - 表格统计：重建误差对比表

### 高级使用

脚本中预留了在不同噪声水平下测试模型性能的框架，您可以进一步拓展以实现更多功能：

- 添加不同噪声类型的对比测试
- 实现多数据集交叉验证
- 增加超参数扫描功能
- 添加模型剪枝和压缩评估

### 结果可视化展示

完成实验后，可以通过以下方式展示结果：

```bash
# 在本地查看结果文件
ls -la ./plots/{实验ID}/

# 使用Python内置HTTP服务器展示结果
python -m http.server 8000
# 然后在浏览器中访问：http://{server-ip}:8000/plots/{实验ID}/
```

通过自动化脚本，研究人员可以轻松地进行系统的对照实验，快速比较不同模型在相同条件下的性能差异，加速研究进展和结果分析。
