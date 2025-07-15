# IABF可视化模块重构方案

## 项目概述

本方案旨在将原有的单文件绘图工具`plot.py`重构为模块化、可扩展的可视化系统。重构后的系统将保留原有的全部功能，同时提高代码的可维护性、可扩展性和复用性。

原始`plot.py`实现了对DynamicIABF、IABF、NECST和UAE等模型的性能可视化，但随着功能增多，单文件结构变得难以维护。此重构方案将功能拆分到多个专注的模块，同时保持与现有代码的兼容性。

## 架构设计

```
项目根目录/
├── plot.py               # 保留原始入口文件，但内部调用visual模块功能
└── visual/
    ├── __init__.py       # 导出主要接口
    ├── README.md         # 本文档，记录重构方案和使用说明
    ├── core.py           # 核心可视化类和公共功能
    ├── data_utils.py     # 数据处理功能
    ├── style_utils.py    # 共享样式设置
    ├── model_utils.py    # 模型识别和查找工具
    ├── plots/
    │   ├── __init__.py
    │   ├── learning.py   # 学习曲线相关图表
    │   ├── distribution.py # 分布相关图表
    │   ├── reconstruction.py # 重建样本相关图表 
    │   ├── metrics.py    # 指标相关图表
    │   └── dynamic_iabf.py # DynamicIABF专用分析
    └── cli.py            # 命令行参数处理
```

这种设计遵循了以下原则：
- **单一职责原则**：每个模块专注于特定功能
- **组合优于继承**：使用组合模式而非深层继承层次
- **兼容性优先**：确保现有代码和调用方式继续有效
- **开放封闭原则**：易于扩展，无需修改现有代码

## 模块职责

### 核心模块

1. **visual/core.py**
   - `ModelVisualizer`类：主可视化类，作为外部接口
   - 内部组合使用各专用可视化器，保持与原接口一致
   - 封装共享配置和资源管理

2. **visual/data_utils.py**
   - `extract_test_data`函数：用于提取和管理测试数据
   - 数据集处理和数据加载相关功能
   - 各种格式数据的规范化处理

3. **visual/style_utils.py**
   - 定义公共的可视化样式
   - 提供出版级别的图表设置
   - 颜色方案、字体和标记样式

4. **visual/model_utils.py**
   - `find_model_dirs`函数：查找模型目录
   - `identify_model_type`函数：识别模型类型
   - 模型元数据解析工具

### 可视化模块

1. **visual/plots/learning.py**
   - `LearningVisualizer`类：绘制学习曲线
   - 训练和验证损失可视化
   - 学习性能比较功能

2. **visual/plots/distribution.py**
   - `DistributionVisualizer`类：绘制分布比较
   - `MutualInformationVisualizer`类：绘制互信息比较
   - 概率分布分析工具

3. **visual/plots/reconstruction.py**
   - `ReconstructionVisualizer`类：绘制重建样本比较
   - `ErrorTableVisualizer`类：绘制重建误差表格
   - 重建质量评估工具

4. **visual/plots/metrics.py**
   - `MetricsVisualizer`类：绘制测试指标比较
   - `NoiseAnalysisVisualizer`类：绘制噪声分析
   - 性能指标比较工具

5. **visual/plots/dynamic_iabf.py**
   - `DynamicIABFVisualizer`类：DynamicIABF特性分析
   - 自适应噪声和渐进式训练分析

### 支持模块

1. **visual/cli.py**
   - `main_cli`函数：处理命令行参数
   - 提供与原始plot.py相同的命令行界面
   - 参数解析和验证

2. **visual/__init__.py**
   - 导出主要接口
   - 版本信息管理
   - 便捷导入支持

## 使用方法

### 作为命令行工具使用

重构后，原有的命令行用法保持不变：

```bash
# 生成所有图表
python plot.py --auto_discover --output_dir ./plots/

# 生成特定类型图表
python plot.py --model_dirs IABF/exp1 UAE/exp1 --plot_type learning
```

### 作为Python库使用

```python
# 导入主可视化类
from visual.core import ModelVisualizer

# 创建可视化器
visualizer = ModelVisualizer(
    base_dir="./results/", 
    model_dirs=["IABF/exp1", "UAE/exp1"],
    dataset_name="mnist"
)

# 生成特定图表
visualizer.plot_learning_curves("./plots/learning_curves.png")

# 生成所有图表
visualizer.plot_all("./plots/")
```

### 使用特定可视化组件

```python
from visual.plots.reconstruction import ReconstructionVisualizer
from visual.data_utils import extract_test_data

# 确保测试数据可用
extract_test_data("mnist")

# 创建特定可视化组件
reconstruction_viz = ReconstructionVisualizer(
    base_dir="./results/", 
    model_dirs=["IABF/exp1"], 
    model_names=["IABF模型"],
    dataset_name="mnist"
)

# 生成可视化
reconstruction_viz.plot("./plots/reconstruction.png")
```

## 扩展指南

### 添加新的可视化类型

1. 在`visual/plots/`下创建新模块文件
2. 实现专用可视化类
3. 在`core.py`中的`ModelVisualizer`类中添加对应方法
4. 更新`cli.py`中的命令行参数支持（如需要）

示例：添加新的参数敏感度分析可视化

```python
# 创建 visual/plots/sensitivity.py
class SensitivityVisualizer:
    def __init__(self, base_dir, model_dirs, model_names, dataset_name):
        # 初始化代码
        pass
        
    def plot(self, save_path=None):
        # 绘图代码
        pass

# 在 core.py 中添加
def plot_parameter_sensitivity(self, save_path=None):
    from visual.plots.sensitivity import SensitivityVisualizer
    visualizer = SensitivityVisualizer(
        self.base_dir, self.model_dirs, self.model_names, self.dataset_name
    )
    return visualizer.plot(save_path)
```

### 支持新的模型类型

1. 在`model_utils.py`中更新`identify_model_type`函数
2. 如需特殊处理，在相关可视化模块中添加支持

## 迁移指南

项目采用了平滑迁移策略，可以分阶段完成：

### 阶段1：基础框架搭建
1. 创建`visual`目录和基本模块结构
2. 实现核心接口和共享功能
3. 保持原始`plot.py`不变

### 阶段2：功能迁移
1. 逐个实现专用可视化模块
2. 更新`plot.py`调用新模块
3. 确保兼容性和功能一致性

### 阶段3：完全迁移
1. `plot.py`完全转为调用新模块的入口文件
2. 全面测试确保功能完整
3. 添加新功能和改进

## 优势和效益

1. **模块化**：每个功能封装在自己的模块中，便于理解和维护
2. **可扩展**：添加新功能不需要修改现有代码
3. **可测试**：每个组件可独立测试
4. **重用性**：可单独使用任何可视化组件
5. **代码质量**：更清晰的结构和更好的组织
6. **兼容性**：保持与现有代码的向后兼容

## 后续工作

1. 单元测试的添加
2. 更详细的API文档
3. 更多交互式可视化选项的添加
4. 对接更多类型的数据源
5. 改进图表样式和配置选项

# IABF可视化模块使用手册

## 项目概述

本模块提供用于可视化和分析DynamicIABF、IABF、NECST和UAE等信息瓶颈模型性能的工具。该系统设计为模块化、可扩展的结构，允许灵活使用各种可视化功能，同时提供友好的命令行和API接口。

系统包含完整的学习曲线分析、分布比较、重建样本评估、测试指标对比以及专门针对DynamicIABF模型的特性分析等功能，支持多模型结果对比和高质量图表导出。

## 目录结构

```
visual/
├── __init__.py       # 主要接口导出
├── README.md         # 本文档，使用说明和文档
├── core.py           # 核心可视化类和主要接口
├── data_utils.py     # 数据处理和加载工具
├── style_utils.py    # 图表样式设置
├── model_utils.py    # 模型识别和查找工具
├── cli.py            # 命令行参数处理
└── plots/            # 专用可视化组件
    ├── __init__.py
    ├── learning.py          # 学习曲线可视化
    ├── distribution.py      # 分布可视化
    ├── mutual_information.py # 互信息可视化
    ├── reconstruction.py    # 重建样本可视化
    ├── metrics.py           # 测试指标可视化
    └── dynamic_iabf.py      # DynamicIABF特性分析
```

## 核心模块详解

### core.py - 核心可视化接口

包含`ModelVisualizer`类，作为整个可视化系统的主要接口。该类组合各专用可视化器，提供一致的API。

**主要功能**:
- 模型结果加载和处理
- 多种图表生成方法
- 图表输出格式控制
- 批量生成一组可视化图表

**使用示例**:
```python
from visual.core import ModelVisualizer

# 创建可视化器
visualizer = ModelVisualizer(
    base_dir="./results/", 
    model_dirs=["IABF/exp1", "NECST/exp1", "UAE/exp1"],
    model_names=["IABF", "NECST", "UAE"],
    dataset_name="mnist"
)

# 生成学习曲线
visualizer.plot_learning_curves("./plots/learning.png")

# 生成所有可视化图表
visualizer.plot_all("./plots/")
```

### data_utils.py - 数据处理工具

提供数据集处理、测试数据提取和加载功能。

**主要功能**:
- `extract_test_data`: 从各数据集提取测试样本
- `load_test_data`: 加载已提取的测试数据
- 数据标准化和预处理

**使用示例**:
```python
from visual.data_utils import extract_test_data, load_test_data

# 提取测试数据
extract_test_data("mnist", num_samples=10)

# 加载测试数据
images, labels = load_test_data("mnist")
```

### model_utils.py - 模型工具

包含模型目录查找、模型类型识别和配置加载工具。

**主要功能**:
- `find_model_dirs`: 自动发现模型结果目录
- `identify_model_type`: 识别模型类型（IABF/NECST/UAE/DynamicIABF）
- `load_config`: 加载模型配置文件

**使用示例**:
```python
from visual.model_utils import find_model_dirs, identify_model_type, load_config

# 查找模型目录
model_dirs, model_names = find_model_dirs("./results/")

# 加载模型配置
config = load_config("IABF/exp1", "./results/")

# 识别模型类型
model_type = identify_model_type(config)
```

### style_utils.py - 样式工具

定义图表样式、颜色、标记和字体等。

**主要功能**:
- `set_publication_style`: 设置出版级图表样式
- `COLORS`, `MARKERS`, `LINE_STYLES`: 一致的样式定义
- 针对不同数据集的参数设置

**使用示例**:
```python
from visual.style_utils import set_publication_style, COLORS

# 设置出版级样式
set_publication_style()

# 使用一致的颜色
import matplotlib.pyplot as plt
plt.plot(x, y, color=COLORS[0])
```

### cli.py - 命令行接口

处理命令行参数并执行相应操作。

**主要功能**:
- `main_cli`: 命令行入口函数
- 参数解析和验证
- 自动发现模型目录和执行可视化

**使用示例**:
```bash
# 命令行中使用
python -m visual.cli --auto_discover --output_dir ./plots/

# 指定特定模型和图表类型
python -m visual.cli --model_dirs IABF/exp1 UAE/exp1 --plot_type learning
```

## 可视化模块详解

### plots/learning.py - 学习曲线可视化

`LearningVisualizer`类提供学习过程的可视化，包括训练和验证损失。

**主要功能**:
- 绘制学习曲线
- 训练和验证损失对比
- 多模型学习表现比较

**使用示例**:
```python
from visual.plots.learning import LearningVisualizer
from visual.core import ModelVisualizer

# 通过核心接口使用
visualizer = ModelVisualizer(...)
visualizer.plot_learning_curves("learning.png")

# 直接使用
learning_viz = LearningVisualizer(visualizer)
learning_viz.plot("learning.png")
```

### plots/distribution.py - 分布可视化

`DistributionVisualizer`类用于比较模型的边缘概率分布。

**主要功能**:
- 绘制激活值分布直方图
- 添加分布统计信息（均值、标准差）
- KDE平滑分布可视化

**使用示例**:
```python
from visual.core import ModelVisualizer

visualizer = ModelVisualizer(...)
visualizer.plot_distribution_comparison("distribution.png")
```

### plots/mutual_information.py - 互信息可视化

包含两个类：
- `MutualInformationVisualizer`: 比较模型间互信息值
- `MutualInformationOverTimeVisualizer`: 展示训练过程中互信息的变化

**主要功能**:
- 互信息对比分析
- 互信息随训练轮次的变化曲线
- 支持对数刻度和平滑处理

**使用示例**:
```python
from visual.core import ModelVisualizer

visualizer = ModelVisualizer(...)
visualizer.plot_mutual_information("mi_comparison.png")
visualizer.plot_mutual_information_over_time("mi_over_time.png", log_scale=True, smoothing=0.3)
```

### plots/reconstruction.py - 重建样本可视化

包含两个类：
- `ReconstructionVisualizer`: 用于比较模型的样本重建效果
- `ErrorTableVisualizer`: 用于生成重建误差指标表格

**主要功能**:
- 原始图像与重建图像对比
- 多模型重建质量比较
- PSNR和SSIM质量指标计算
- 颜色编码的误差指标表格

**使用示例**:
```python
from visual.core import ModelVisualizer

visualizer = ModelVisualizer(...)
# 比较重建样本
visualizer.plot_reconstruction_samples("reconstruction.png", num_samples=5)
# 生成误差指标表格
visualizer.plot_reconstruction_errors_table("errors_table.png")
# 收集并导出误差数据到CSV
df = visualizer.collect_reconstruction_errors("errors.csv")
```

### plots/metrics.py - 测试指标可视化

包含两个类：
- `MetricsVisualizer`: 比较不同模型的测试性能指标
- `NoiseAnalysisVisualizer`: 分析不同噪声水平下的模型性能

**主要功能**:
- 测试指标条形图比较
- 多指标对比分析（损失、准确率、互信息等）
- 噪声鲁棒性分析曲线
- 测试指标导出为CSV

**使用示例**:
```python
from visual.core import ModelVisualizer

visualizer = ModelVisualizer(...)
# 测试指标比较
visualizer.plot_test_metrics("metrics.png")
# 噪声分析
visualizer.plot_noise_analysis("noise_analysis.png")
# 导出测试指标
visualizer.export_test_metrics("metrics.csv")
```

### plots/dynamic_iabf.py - DynamicIABF特性分析

`DynamicIABFVisualizer`类用于分析DynamicIABF模型的特有特性。

**主要功能**:
- β值适应过程分析
- 噪声水平适应过程分析
- 信息平面分析（互信息与噪声/β值的关系）
- 参数适应综合摘要

**使用示例**:
```python
from visual.core import ModelVisualizer

visualizer = ModelVisualizer(...)
# 生成DynamicIABF特性综合分析
visualizer.plot_dynamic_iabf_features("dynamic_features.png")

# 直接使用更详细控制
from visual.plots.dynamic_iabf import DynamicIABFVisualizer
dynamic_viz = DynamicIABFVisualizer(visualizer)
# β值适应分析
dynamic_viz.plot_beta_adaptation("beta.png")
# 噪声适应分析
dynamic_viz.plot_noise_adaptation("noise.png")
# 信息平面分析
dynamic_viz.plot_information_plane("info_plane.png")
# 适应过程综合摘要
dynamic_viz.plot_adaptation_summary("adaptation.png")
```

## 使用方法

### 基本使用流程

1. **初始化可视化器**:
```python
from visual.core import ModelVisualizer

visualizer = ModelVisualizer(
    base_dir="./results/",  # 结果文件的基础目录
    model_dirs=["IABF/exp1", "NECST/exp2"],  # 模型结果目录
    model_names=["IABF模型", "NECST模型"],  # 图例中显示的名称
    dataset_name="mnist"  # 数据集名称
)
```

2. **生成特定图表**:
```python
# 生成学习曲线
visualizer.plot_learning_curves("./plots/learning.png")

# 生成互信息比较
visualizer.plot_mutual_information("./plots/mi.png")

# 生成重建样本比较
visualizer.plot_reconstruction_samples("./plots/reconstruction.png", num_samples=5)
```

3. **生成所有图表**:
```python
# 一次性生成所有支持的图表
visualizer.plot_all("./plots/")
```

### 自动发现模型目录

可以使用`find_model_dirs`函数自动发现模型目录:

```python
from visual.model_utils import find_model_dirs
from visual.core import ModelVisualizer

# 自动发现所有模型目录
model_dirs, model_names = find_model_dirs("./results/")

# 仅发现特定类型的模型
iabf_dirs, iabf_names = find_model_dirs("./results/", model_types=["IABF"])

# 使用发现的目录创建可视化器
visualizer = ModelVisualizer(
    base_dir="./results/",
    model_dirs=model_dirs,
    model_names=model_names
)
```

### 命令行使用

可以使用命令行工具快速生成图表:

```bash
# 自动发现模型并生成所有图表
python -m visual.cli --auto_discover --output_dir ./plots/

# 只生成特定类型的图表
python -m visual.cli --model_dirs IABF/exp1 UAE/exp1 --plot_type learning

# 生成DynamicIABF特性分析
python -m visual.cli --model_dirs DynamicIABF/exp1 --plot_type dynamic_iabf

# 指定数据集
python -m visual.cli --auto_discover --dataset mnist --output_dir ./plots/
```

可用的`plot_type`选项:
- `learning` - 学习曲线
- `distribution` - 分布比较
- `mutual_information` - 互信息比较
- `mi_over_time` - 互信息随时间变化
- `reconstruction` - 重建样本比较
- `errors_table` - 重建误差表格
- `metrics` - 测试指标比较
- `noise_analysis` - 噪声分析
- `dynamic_iabf` - DynamicIABF特性分析
- `all` - 所有图表（默认）

### 使用plot.py（向后兼容）

原始的`plot.py`脚本保持对旧代码的兼容性，但内部使用新的模块化系统:

```bash
# 使用原始plot.py
python plot.py --auto_discover --output_dir ./plots/
```

## 扩展指南

### 添加新的可视化类型

1. 在`visual/plots/`下创建新模块文件
2. 实现专用可视化类，接收`model_visualizer`作为参数
3. 在`core.py`中的`ModelVisualizer`类中添加对应方法
4. 更新`cli.py`中的命令行参数支持（如需要）

示例：添加散点图可视化

```python
# 创建 visual/plots/scatter.py
class ScatterVisualizer:
    def __init__(self, model_visualizer):
        self.model_visualizer = model_visualizer
        self.base_dir = model_visualizer.base_dir
        self.model_dirs = model_visualizer.model_dirs
        self.model_names = model_visualizer.model_names
        
    def plot(self, save_path=None):
        # 实现绘图逻辑
        pass

# 在 core.py 中添加
def plot_feature_scatter(self, save_path=None):
    from .plots.scatter import ScatterVisualizer
    visualizer = ScatterVisualizer(self)
    return visualizer.plot(save_path)
```

### 故障排除

1. **缺少数据文件**:
   - 确保模型目录中包含必要的日志和数据文件（log.txt, metrics.pkl等）
   - 使用`extract_test_data`确保测试数据可用

2. **图表不显示数据**:
   - 检查模型目录路径是否正确
   - 验证日志文件中是否包含所需指标

3. **样式问题**:
   - 使用`set_publication_style()`确保样式一致
   - 检查matplotlib版本是否兼容

## 高级功能

### 自定义图表样式

```python
import matplotlib.pyplot as plt
from visual.style_utils import set_publication_style, COLORS

# 设置基本样式
set_publication_style()

# 自定义额外样式
plt.rcParams.update({
    'font.size': 14,
    'figure.figsize': (10, 8)
})

# 创建可视化器并生成图表
visualizer = ModelVisualizer(...)
visualizer.plot_learning_curves("learning_custom.png")
```

### 导出数据进行外部分析

```python
from visual.core import ModelVisualizer

visualizer = ModelVisualizer(...)

# 导出测试指标到CSV
metrics_df = visualizer.export_test_metrics("metrics.csv")

# 收集重建误差指标
errors_df = visualizer.collect_reconstruction_errors("errors.csv")

# 外部分析（如使用pandas）
import pandas as pd
combined_df = pd.merge(metrics_df, errors_df, on="Model")
```

## API参考

每个主要类和函数的详细参数说明可以在各模块的文档字符串中找到。主要接口包括:

- `ModelVisualizer` - 主可视化类
- `LearningVisualizer` - 学习曲线可视化
- `DistributionVisualizer` - 分布可视化
- `MutualInformationVisualizer` - 互信息比较
- `ReconstructionVisualizer` - 重建样本比较
- `MetricsVisualizer` - 测试指标比较
- `NoiseAnalysisVisualizer` - 噪声分析
- `DynamicIABFVisualizer` - DynamicIABF特性分析
- `find_model_dirs` - 模型目录查找
- `extract_test_data` - 测试数据提取
