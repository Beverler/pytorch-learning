# PyTorch Learning

记录从 PyTorch 基础、CNN 图像分类，到 Transformer 原理与 Hugging Face Transformers 实践的学习过程。

当前仓库包含 **CIFAR-10 图像分类训练代码、单张图片预测代码，以及 Transformers 一周学习路线**。适合已经接触 Python，希望通过阅读、运行和修改代码理解深度学习流程的学习者。

**学习入口：[CNN 训练代码](./train_gpu.py) · [单张图片预测](./test.py) · [Transformers 一周学习路线](./Transformers-%E4%B8%80%E5%91%A8%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md)**

## 仓库内容

```text
pytorch-learning/
├── .gitattributes
├── README.md
├── train_gpu.py                   # CIFAR-10：训练、测试、TensorBoard、模型保存
├── test.py                        # 加载本地模型，预测单张图片
└── Transformers-一周学习路线.md    # 每天的官方文档、GitHub 示例和实践目标
```

| 文件 | 主要内容 | 建议关注的知识点 |
| --- | --- | --- |
| [train_gpu.py](./train_gpu.py) | 定义 Beverly CNN，训练并测试 CIFAR-10 分类模型 | DataLoader、卷积、最大池化、线性层、交叉熵、SGD、GPU |
| [test.py](./test.py) | 图片预处理、模型加载、前向推理、类别预测 | RGB、Resize、批次维度、eval、no_grad、argmax |
| [Transformers-一周学习路线.md](./Transformers-%E4%B8%80%E5%91%A8%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md) | 七天的阅读与复现安排 | Pipeline、Tokenizer、Attention、微调、自定义中文分类项目 |

Transformers 部分目前为学习计划与参考链接，配套训练脚本可按照路线逐步补充。

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Beverler/pytorch-learning.git
cd pytorch-learning
```

### 2. 准备环境

建议使用独立的 Python 环境，例如通过 Anaconda 创建 Python 3.11 环境：

```bash
conda create -n pytorch-learning python=3.11 -y
conda activate pytorch-learning
```

先根据操作系统和 CPU/CUDA 平台，使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装相互匹配的 `torch` 与 `torchvision`，再安装以下依赖：

```bash
python -m pip install tensorboard pillow
```

检查当前环境与 GPU 是否可用：

```bash
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: False` 时，当前训练脚本会使用 CPU。使用 PyCharm 时，请确认项目解释器选择了安装这些依赖的环境。

### 3. 启动训练

在仓库根目录运行：

```bash
python train_gpu.py
```

脚本将自动下载 CIFAR-10，训练 10 轮，每轮完成测试并保存模型。以下参数对应当前代码：

| 配置 | 当前值 |
| --- | --- |
| 数据集 | CIFAR-10 |
| 输入 | RGB，`3 × 32 × 32` |
| 分类数 | 10 |
| batch size | 64 |
| 训练轮数 | 10，循环为 `range(10)` |
| 损失函数 | `nn.CrossEntropyLoss()` |
| 优化器 | SGD |
| 学习率 | `1e-3`，即 `0.001` |
| 设备 | CUDA 可用时使用 GPU，否则使用 CPU |
| 模型文件 | `Beverly_0.pth` ～ `Beverly_9.pth` |

`epoch = 10` 虽然在脚本中定义，但实际训练轮数由 `for i in range(10)` 控制。修改轮数时需要检查实际循环。

### 4. 查看 TensorBoard

保持工作目录为仓库根目录，另开终端运行：

```bash
tensorboard --logdir ../logs/logs_modeltrain
```

打开终端提示的网址，即可查看 `train_loss`、`test_loss` 与 `test_accuracy`。训练损失每 100 个训练批次记录一次，测试指标每轮记录一次。

## 路径与生成文件

当前脚本的相对路径以**程序运行时的工作目录**为基准。按上述方式从仓库根目录运行时，文件位置如下：

| 内容 | 代码中的路径 | 实际位置 |
| --- | --- | --- |
| CIFAR-10 数据集 | `../data/dataset` | 仓库上一级目录中的 `data/dataset` |
| TensorBoard 日志 | `../logs/logs_modeltrain` | 仓库上一级目录中的 `logs/logs_modeltrain` |
| 待预测图片 | `../imgs/02.jpg` | 仓库上一级目录中的 `imgs/02.jpg` |
| 训练生成的模型 | `Beverly_0.pth` 等 | 当前工作目录 |

仓库没有附带训练权重和示例图片。运行预测前，需要先生成模型文件，并准备自己的图片。PyCharm 的 Working directory 建议设置为仓库根目录。

## CNN 模型结构

`Beverly` 继承 `nn.Module`，使用 `nn.Sequential` 串联卷积、最大池化、展平与线性层。

其中 `B` 表示批次中的图片数量：

| 层 | 配置 | 输出形状 |
| --- | --- | --- |
| 输入 | RGB 图片 | `[B, 3, 32, 32]` |
| 卷积 1 | `Conv2d(3, 32, 5, 1, 2)` | `[B, 32, 32, 32]` |
| 最大池化 1 | `MaxPool2d(2)` | `[B, 32, 16, 16]` |
| 卷积 2 | `Conv2d(32, 32, 5, 1, 2)` | `[B, 32, 16, 16]` |
| 最大池化 2 | `MaxPool2d(2)` | `[B, 32, 8, 8]` |
| 卷积 3 | `Conv2d(32, 64, 5, 1, 2)` | `[B, 64, 8, 8]` |
| 最大池化 3 | `MaxPool2d(2)` | `[B, 64, 4, 4]` |
| 展平 | `Flatten()` | `[B, 1024]` |
| 线性层 1 | `Linear(1024, 64)` | `[B, 64]` |
| 线性层 2 | `Linear(64, 10)` | `[B, 10]` |

`1024 = 64 × 4 × 4`，因此展平后可以输入第一个线性层。最终的 10 个数是各类别的原始分数（logits），不是概率；训练时直接交给交叉熵，预测时通过 `argmax` 取分数最高的类别。

当前模型没有显式加入 ReLU 等激活层，两个线性层之间也没有激活函数。后续可以增加激活层，与当前结构比较训练表现。

## 单张图片预测

### 1. 准备图片与模型

先完成训练，并准备一张图片。当前 [test.py](./test.py) 默认配置为：

```python
images_path = "../imgs/02.jpg"
model = torch.load(
    "Beverly_GPU_29.pth",
    map_location=torch.device("cpu"),
    weights_only=False,
)
```

**训练脚本生成的名称与测试脚本默认名称不同。** 如果使用当前训练脚本最后一轮保存的模型，请在自己的 `test.py` 中将加载部分设为：

```python
model = torch.load(
    "Beverly_9.pth",
    map_location=torch.device("cpu"),
    weights_only=False,
)
```

同时将 `images_path` 设置为实际图片路径。

### 2. 执行预测

```bash
python test.py
```

图片依次经过 `RGB → Resize(32, 32) → ToTensor → [1, 3, 32, 32]`，再送入模型。推理使用 `model.eval()` 和 `torch.no_grad()`，输出形状为 `[1, 10]`。

当前脚本最终打印类别编号。CIFAR-10 类别顺序如下，可在得到 `output` 后按需添加类别名称输出：

```python
classes = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
predicted_id = output.argmax(dim=1).item()
print("预测类别：", classes[predicted_id])
```

对应中文为：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。输入图片应尽量属于这 10 类；模型会为任何输入选择一个类别。

当前采用保存整个模型对象的方式，因此加载时保留了 `Beverly` 类定义；加载文件应使用自己训练或来源可信的模型。后续可参考 [PyTorch 官方保存与加载教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)，改用 `state_dict` 保存参数。

## 训练过程中的知识点

训练的主要流程：

```text
model.train()
→ 前向传播
→ 计算交叉熵损失
→ optimizer.zero_grad()
→ loss.backward()
→ optimizer.step()
```

测试时使用 `model.eval()` 与 `torch.no_grad()`，通过 `output.argmax(1)` 得到预测编号，再统计预测正确的样本数。

阅读代码时重点区分：

- **epoch 与 step**：一轮遍历完整训练集；一次训练 step 处理一个批次。
- **反向传播与参数更新**：`backward()` 计算梯度，`optimizer.step()` 才更新参数。
- **训练与评估模式**：`train()`、`eval()` 设置模块模式，是否记录梯度还需要分别控制。
- **损失与准确率**：当前 `total_test_loss` 是各批平均损失的累加，不是整个测试集按样本计算的平均损失；准确率为正确样本数除以测试样本总数。
- **模型与数据设备**：进行运算的模型、图片和标签需要放在相同设备上。

## 下一阶段：Transformers 一周学习路线

在完成 CNN 训练与推理之后，按 [完整学习路线](./Transformers-%E4%B8%80%E5%91%A8%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md) 继续学习。每天约 2～3 小时，包含明确的官方文档、GitHub 文件、复现步骤和验收目标。

| 日期 | 学习主题 | 实践成果 |
| --- | --- | --- |
| 第 1 天 | Pipeline 与预训练模型 | 英文评论情感分类器 |
| 第 2 天 | Tokenizer、输入张量与 logits | 手动复现 Pipeline |
| 第 3 天 | Attention 与 Transformer 结构 | 注意力计算与热力图 |
| 第 4 天 | Datasets 与 Trainer | 微调 IMDb 影评分类模型 |
| 第 5 天 | PyTorch 手写训练循环 | 改写第 4 天的训练过程 |
| 第 6 天 | 中文模型与自定义数据 | 中文酒店评论分类器 |
| 第 7 天 | 保存、加载与独立预测 | 整理成可复现的独立项目 |

Transformer 是一种模型结构；Transformers 是 Hugging Face 的模型工具库。已有的 PyTorch 训练、损失函数、反向传播与 GPU 知识仍然适用。

## 后续完善方向

- [ ] 统一训练与推理的模型路径，支持命令行传入图片和模型。
- [ ] 增加训练集随机打乱、独立验证集与随机种子设置。
- [ ] 按样本数记录平均测试损失，保存表现最好的模型。
- [ ] 整理 `requirements.txt` 与实际运行环境。
- [ ] 对比加入 ReLU 前后的 CNN 表现。
- [ ] 完成 Transformers 官方示例复现，补充中文分类项目代码与实验结果。

## 官方参考资料

- [PyTorch 安装](https://pytorch.org/get-started/locally/)
- [PyTorch 基础教程](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Torchvision CIFAR-10 数据集](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)
- [PyTorch 模型保存与加载](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face 官方课程](https://huggingface.co/learn/llm-course/chapter1/1)
- [Hugging Face 官方 Notebook](https://github.com/huggingface/notebooks)
