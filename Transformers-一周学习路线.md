# Transformers 一周学习路线：从 PyTorch 到自己的中文分类项目

## 学习目标与时间

你已经学习过 CNN、DataLoader、损失函数、反向传播、优化器、GPU 训练和模型保存，可以直接进入预训练 Transformer 的调用与微调。

- 建议每天 2～3 小时，约 14～21 小时；模型下载和实际训练时间取决于网络及设备。
- 一周目标：理解注意力的基本计算，复现官方文本分类示例，再独立完成中文评论分类项目。
- Transformer 指模型结构；Transformers 指 Hugging Face 提供的模型工具库。本路线同时覆盖基础原理和库的使用。
- 每天按“读文档 → 运行参考代码 → 自己修改 → 检查结果”的顺序学习。

## 七天总览

| 日期 | 主题 | 当天小项目 | 当天成果 |
| --- | --- | --- | --- |
| 第 1 天 | Pipeline 与预训练模型 | 英文评论情感分类器 | 输入多条评论，输出类别与分数 |
| 第 2 天 | Tokenizer 与模型输出 | 拆解并手动复现 Pipeline | 展示 tokens、输入张量、logits 和预测 |
| 第 3 天 | Attention 与 Transformer 结构 | 注意力计算与热力图 | 手写注意力并与 PyTorch 结果对比 |
| 第 4 天 | Datasets 与 Trainer | IMDb 影评分类微调 | 跑通训练、验证、保存 |
| 第 5 天 | 手写训练循环 | 用 PyTorch 重写第 4 天训练过程 | 理解 Trainer 封装了什么 |
| 第 6 天 | 中文模型与自定义数据 | 中文酒店评论分类 | 完成数据清洗、划分、微调、评估 |
| 第 7 天 | 保存、加载与独立推理 | 整理成自己的项目 | 不重新训练即可预测，附完整 README |

以下 GitHub 链接指向具体文件。第 4～7 天逐步改造同一个分类项目。

## 第 1 天：用 Pipeline 完成第一次推理

### 官方学习文档

- [Transformers, what can they do?：Transformers 能做什么](https://huggingface.co/learn/llm-course/chapter1/3)

重点阅读 `Working with pipelines` 和文本分类示例。

### GitHub 参考代码

- [Hugging Face 官方 Notebook：chapter1/section3.ipynb](https://github.com/huggingface/notebooks/blob/main/course/en/chapter1/section3.ipynb)

### 小项目：英文评论情感分类器

1. 运行 Notebook 中的 `sentiment-analysis` 示例。
2. 输入 10 条自己写的英文正面、负面评论，尝试批量预测。
3. 查看 `label` 与 `score`，记录至少一个判断不符合预期的例子。
4. 保存成自己的 `day01_pipeline.py`，显式写出使用的模型名称。

验收：能够独立解释“预处理 → 模型 → 后处理”的过程，知道使用已经训练好的模型属于推理。

## 第 2 天：拆开 Pipeline，理解文本如何进入模型

### 官方学习文档

- [Behind the pipeline：Pipeline 背后的过程](https://huggingface.co/learn/llm-course/chapter2/2)
- [Tokenizers：分词器](https://huggingface.co/learn/llm-course/chapter2/4)

重点掌握 `AutoTokenizer`、`input_ids`、`attention_mask`、padding、truncation 和 `logits`。

### GitHub 参考代码

- [官方 Notebook：chapter2/section2_pt.ipynb](https://github.com/huggingface/notebooks/blob/main/course/en/chapter2/section2_pt.ipynb)
- [官方 Notebook：chapter2/section4_pt.ipynb](https://github.com/huggingface/notebooks/blob/main/course/en/chapter2/section4_pt.ipynb)

### 小项目：手动复现情感分类 Pipeline

1. 使用与第 1 天相同的 checkpoint 加载 Tokenizer 和 `AutoModelForSequenceClassification`。
2. 打印 tokens、`input_ids`、`attention_mask` 及其形状。
3. 用 `model(**inputs)` 获取 `outputs.logits`。
4. 对最后一维使用 softmax，取得预测类别及概率，与 Pipeline 结果对比。
5. 对不同长度的两句话进行批量预测，观察 padding 的位置。

验收：能够说明 `[B, L]` 表示一批 token 编号、`[B, C]` 表示每个样本的类别分数；理解基础模型的隐藏状态与分类模型的 logits 不同。

## 第 3 天：理解 Attention，并做一个小实验

### 官方学习文档

- [How do Transformers work?：Transformer 如何工作](https://huggingface.co/learn/llm-course/chapter1/4)
- [PyTorch 官方教程：Scaled Dot Product Attention](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)

先读架构、注意力和 Encoder/Decoder 的说明；PyTorch 教程重点运行开头的 `Example Usage`。性能测试、FlashAttention 和编译部分留作后续选读。

### GitHub 参考代码

- [PyTorch 官方源码：scaled_dot_product_attention_tutorial.py](https://github.com/pytorch/tutorials/blob/main/intermediate_source/scaled_dot_product_attention_tutorial.py)

### 小项目：注意力矩阵与热力图

1. 构造很小的随机 `Q`、`K`、`V`，例如形状均为 `[1, 4, 8]`。
2. 手写以下公式，不加 mask，关闭 dropout：

   `attention = softmax(Q @ K.transpose(-2, -1) / sqrt(d_k), dim=-1)`

   `output = attention @ V`

3. 与 `F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)` 的输出比较，检查数值是否接近。
4. 用 Matplotlib 画出 `[4, 4]` 的注意力权重热力图。
5. 了解多头注意力、位置信息、线性层、残差连接与 LayerNorm 在结构中的作用。

验收：能够解释 Q/K/V 的作用，知道热力图每一行对应一个 query 对各个 key 的关注权重。随机张量实验用于理解计算，没有真实语言语义。

## 第 4 天：复现官方 IMDb 影评分类微调

### 官方学习文档

- [Text classification：文本分类完整教程](https://huggingface.co/docs/transformers/tasks/sequence_classification)

按 `Load IMDb dataset → Preprocess → Evaluate → Train` 的顺序阅读。

### GitHub 参考代码

- [官方项目 Notebook：sequence_classification.ipynb](https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/sequence_classification.ipynb)

这个示例使用 DistilBERT 对 IMDb 影评进行正负面分类，包含数据处理、微调、评估和推理。

### 小项目：微调自己的英文影评分类器

1. 先运行数据加载与预处理，检查 `text`、`label` 字段。
2. 从原始训练集中随机抽取约 1000 条用于训练，再划出约 200 条作为验证集；先 shuffle 再抽样。
3. 使用示例模型 `distilbert/distilbert-base-uncased`，先尝试 `max_length=128`、batch size 8、1 个 epoch。
4. 学习 `TrainingArguments`、`Trainer`、`DataCollatorWithPadding` 和 `compute_metrics`。
5. 记录训练损失、验证准确率和参数设置。
6. 首次在本地保存即可，把示例中的 `push_to_hub` 设为 `False`，跳过登录和上传。

验收：完成一次训练、一次验证，并用训练后的模型预测新影评。小样本主要用于跑通流程，不设固定准确率目标。

## 第 5 天：把 Trainer 改写成你熟悉的 PyTorch 训练循环

### 官方学习文档

- [A full training loop：完整训练循环](https://huggingface.co/learn/llm-course/chapter3/4)

### GitHub 参考代码

- [官方 Notebook：chapter3/section4.ipynb](https://github.com/huggingface/notebooks/blob/main/course/en/chapter3/section4.ipynb)

注意：参考 Notebook 原任务是 MRPC 句子对分类。今天借鉴它的训练循环，将数据输入改为第 4 天 IMDb 的单句 `text`，沿用第 4 天的模型和数据划分。

### 小项目：手写影评分类训练脚本

1. 用处理好的 IMDb 数据创建 DataLoader，移除模型不接受的原始文本列。
2. 检查 batch 中有 `input_ids`、`attention_mask`、`labels`；手写数据处理时注意把 `label` 统一为 `labels`。
3. 将模型和 batch 中的张量放到相同设备。
4. 使用 `outputs = model(**batch)` 和 `loss = outputs.loss`。
5. 完成 `zero_grad → backward → step`，记录每批损失。
6. 用 `model.eval()` 和 `torch.no_grad()` 计算验证准确率。

验收：能够把 Transformer 训练循环与自己之前的 CNN 训练代码逐项对应；知道 Trainer 封装了数据迭代、反向传播、参数更新、评估和保存等过程。

## 第 6 天：改造成中文酒店评论分类项目

### 官方学习文档

- [Datasets：加载本地 CSV](https://huggingface.co/docs/datasets/loading#csv)
- [Text classification：复用文本分类训练流程](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [中文预训练模型说明：google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)

### GitHub 参考项目与数据

- [训练代码参考：官方 sequence_classification.ipynb](https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/sequence_classification.ipynb)
- [中文数据集：ChineseNlpCorpus / ChnSentiCorp_htl_all](https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/ChnSentiCorp_htl_all)
- [数据文件：ChnSentiCorp_htl_all.csv](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv)
- [数据字段与标签说明：intro.ipynb](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/intro.ipynb)

前一项是官方训练示例，后三项是第三方中文数据资源。

### 小项目：中文酒店评论正负面分类

1. 读取 CSV：原始文本列为 `review`，标签列为 `label`；`0` 为负面，`1` 为正面。
2. 删除空文本和重复样本，将 `review` 重命名为 `text`。
3. 固定随机种子，划分训练、验证、测试集，例如 80%/10%/10%，尽量保持类别比例。
4. 同时替换 Tokenizer 和模型 checkpoint 为 `google-bert/bert-base-chinese`，分类类别数设为 2。
5. 复用第 4 天 Trainer 或第 5 天手写循环，先用小规模数据和 1 个 epoch 跑通。
6. 保存模型，记录 accuracy、F1，并阅读至少 10 条预测错误的评论。

验收：能够说明自己修改了数据读取、文本字段、模型、标签映射、评估流程中的哪些内容；能够预测自己输入的中文酒店评论。

注意：该 BERT checkpoint 是预训练基础模型。添加分类头后，需要进行分类微调，才能把输出用于你的情感分类任务。

## 第 7 天：保存、重新加载，并整理成独立项目

### 官方学习文档

- [Models：重点阅读 Loading and saving](https://huggingface.co/learn/llm-course/chapter2/3)
- [Text classification：重点阅读 Inference](https://huggingface.co/docs/transformers/tasks/sequence_classification#inference)

### GitHub 参考代码

- [官方 Notebook：chapter2/section3_pt.ipynb](https://github.com/huggingface/notebooks/blob/main/course/en/chapter2/section3_pt.ipynb)
- [官方分类 Notebook：Inference 部分](https://github.com/huggingface/notebooks/blob/main/transformers_doc/en/sequence_classification.ipynb)

### 小项目：可以独立运行的中文评论分类系统

1. 用 `save_pretrained()` 同时保存模型和 Tokenizer。
2. 新建 `predict.py`，从本地目录加载模型与 Tokenizer，不依赖训练脚本里的变量。
3. 输入一句中文评论，输出“正面/负面”、预测概率。
4. 检查保存前后的模型在评估模式下，对同一输入的预测是否一致。
5. 用 `evaluate.py` 在独立测试集上完成最终评估。
6. 编写 README：任务、数据来源、安装方法、训练命令、预测命令、指标、错误案例。
7. 记录实际运行成功的依赖版本，使以后可以复现。

建议最终目录：

```text
chinese-review-classifier/
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── README.md
└── saved_model/          # 本地生成的模型与 Tokenizer
```

验收：打开新的 Python 进程，仅加载 `saved_model` 就能预测；别人按 README 能重现你的操作。README 中标明参考的官方代码与数据来源，模型权重可先保留在本地。

## 复现时最需要注意的地方

1. Tokenizer 与 checkpoint 配套；更换模型时，检查输入字段和模型输出。
2. 分类模型输出的 logits 是分数。训练时直接用于交叉熵，展示概率时再做 softmax。
3. 模型和输入张量位于相同设备；手写推理时保留 `eval()` 和 `no_grad()`。
4. 验证集用于调参，测试集留到最终评估。官方 IMDb 示例用 test 进行训练期间评估；自己的项目建议另划验证集。
5. 中文数据类别数量不均衡时，同时关注 F1、各类别表现和错误案例。
6. 官方 Notebook 可能随版本更新。记录自己跑通的版本；参数不匹配时，对照该版本的文档检查。
7. 显存不足时，先减小 batch size 和 max_length。每天建议用时不包含下载等待，也不保证固定训练速度。

## 一周结束后的自检

- [ ] 能解释 Transformer 结构与 Transformers 库的区别。
- [ ] 能解释 Tokenizer、attention_mask、logits。
- [ ] 能手写简单的缩放点积注意力，并检查输出形状。
- [ ] 能复现官方文本分类微调示例。
- [ ] 能手写训练、验证循环。
- [ ] 能换用中文数据集和中文模型。
- [ ] 能保存、加载并进行独立推理。
- [ ] 有一个写明运行方法、指标和参考来源的项目。
