# 数据集格式和模型返回值分析

## 1. 数据集格式分析

### 处理前后的数据集格式

| 数据集 | 处理前格式 | 处理后格式 | 变换函数 | 备注 |
| ----- | --------- | --------- | ------- | ---- |
| mnist | TFRecords (uint8) | float32 [batch_size, 784] | _preprocess_mnist | 归一化到[0,1] |
| BinaryMNIST | TFRecords (float32) | float32 [batch_size, 784] | _preprocess_binary_mnist | 二值图像数据 |
| random | TFRecords (float32) | float32 [batch_size, 100] | _preprocess_binary_mnist | 随机生成的二进制数据 |
| omniglot | TFRecords (float64) | float32 [batch_size, 784] | _preprocess_omniglot | 转换为float32 |
| binary_omniglot | TFRecords (float64) | float32 [batch_size, 784] | _preprocess_omniglot | 二值化的Omniglot数据 |
| svhn | TFRecords (uint8) | float32 [batch_size, 32, 32, 3] | _preprocess_svhn | 归一化到[0,1] |
| cifar10 | TFRecords (uint8) | float32 [batch_size, 32, 32, 3] | _preprocess_cifar10 | CHW转为HWC格式，归一化到[0,1] |
| celebA | TFRecords (uint8) | float32 [batch_size, 64, 64, 3] | _preprocess_celebA | 归一化到[0,1] |

## 2. 模型训练过程中的重要变量和返回值

### necst.py 中的重要变量

| 变量名 | 格式 | 描述 | 来源函数 |
| ----- | ---- | ---- | ------- |
| mean_ | float32 [batch_size, n_bits] | 编码器输出的概率分布 | create_collapsed_computation_graph |
| z | float32 [batch_size, n_bits] | 采样的二进制隐变量 | create_collapsed_computation_graph |
| classif_z | float32 [batch_size, n_bits] | 分类器使用的隐变量表示 | create_collapsed_computation_graph |
| q | Bernoulli/RelaxedBernoulli | 采样分布 | create_collapsed_computation_graph |
| x_reconstr_logits | float32 [batch_size, input_dim] | 重构输出的logits | create_collapsed_computation_graph |
| mean | float32 [batch_size, n_bits] | 最终的编码器输出 | create_collapsed_computation_graph |
| theta_loss | float32 | 解码器的损失函数值 | vimco_loss |
| phi_loss | float32 | 编码器的损失函数值 | vimco_loss |
| reconstr_loss | float32 | 重构损失 | vimco_loss |
| conditional_entropy | float32 | 条件熵 | get_conditional_entropy |
| kl_loss | float32 | KL散度损失 | kl_in_batch |
| denoising_loss | float32 | 去噪损失 | denoising_decoder |
| tc_loss | float32 | 总相关性损失 | 基于判别器输出计算 |
| D_loss | float32 | 判别器损失 | 基于判别器输出和标签计算 |

### main.py 中的重要变量和返回值

| 变量名 | 格式 | 描述 | 来源 |
| ----- | ---- | ---- | ---- |
| learning_curves | dict | 训练过程中的损失曲线 | model.train() |
| best_ckpt | string | 最佳检查点路径 | model.train() |
| output_path | string | 模型输出路径 | create_output_path() |

### utils.py 中的实用函数

| 函数名 | 输入 | 输出 | 描述 |
| ----- | ---- | ---- | ---- |
| load_dynamic | class_name, module_name | class | 动态加载类 |
| get_activation_fn | activation | function | 获取激活函数 |
| get_optimizer_fn | optimizer | function | 获取优化器函数 |
| plot | samples, m, n, px, title | figure | 绘制样本图像 |
| provide_data | data, batch_size | iterator, stats | 提供批量数据和统计信息 |
| provide_unlabelled_data | data, batch_size | iterator | 提供无标签批量数据 |

### datasource.py 中的数据集处理

| 函数名 | 输入 | 输出 | 描述 |
| ----- | ---- | ---- | ---- |
| get_tf_dataset | split | tf.data.Dataset | 获取TFRecord格式数据集 |
| get_binary_tf_dataset | split | tf.data.Dataset | 获取二值化TFRecord数据集 |
| get_cifar10_tf_dataset | split | tf.data.Dataset | 获取CIFAR10格式数据集 |
| get_tf_dataset_celebA | split | tf.data.Dataset | 获取CelebA格式数据集 |

## 3. 模型训练过程

### NECST/IABF/DynamicIABF 训练过程中的重要步骤

| 阶段 | 主要操作 | 输入 | 输出 | 函数 |
| --- | ------- | ---- | ---- | ---- |
| 初始化 | 创建计算图 | datasource | 模型实例 | __init__ |
| 编码 | 将输入转换为隐变量 | x [batch_size, input_dim] | mean, z [batch_size, n_bits] | encoder |
| 噪声通道 | 向隐变量添加噪声 | z [batch_size, n_bits] | corrupted z [batch_size, n_bits] | perturb_latent_code |
| 解码 | 从隐变量重构输入 | z [batch_size, n_bits] | x_reconstr_logits [batch_size, input_dim] | decoder |
| 损失计算 | 计算各种损失函数 | x, x_reconstr_logits | theta_loss, phi_loss, reconstr_loss | vimco_loss |
| 优化 | 更新模型参数 | 损失值 | 优化后的参数 | train |
| 评估 | 在测试集上评估模型 | 测试数据 | test_loss | test |
| 重构 | 生成重构样本 | 测试数据 | 重构图像 | reconstruct |
| MCMC采样 | 执行马尔可夫链采样 | 起始样本 | 采样轨迹 | markov_chain |

### 动态模型特有的功能

| 功能 | 描述 | 实现函数 |
| --- | ----- | ------- |
| 自适应噪声调整 | 根据损失动态调整噪声水平 | adaptive_noise_level |
| 渐进式训练 | 逐步增加训练难度 | progressive_training_noise |
