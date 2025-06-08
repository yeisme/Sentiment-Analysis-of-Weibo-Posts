# Sentiment Analysis of Weibo Posts During the Pandemic - 疫情微博情绪分类

## 任务描述

输出是该微博所蕴含的情绪类别。微博情绪分类任务旨在识别微博中蕴含的情绪，输入是一条微博，在本次任务中，我们将微博按照其蕴含的情绪分为以下六个类别之一: 积极、愤怒、悲伤、恐惧、惊奇和无情绪 (0,1,2,3,4,5)。

> 数据集
> 疫情微博训练数据集包括 6,606 条微博，测试数据集包含 5.000 条微博
> 训练数据集: virus_train.csv
> 测试数据集: virus_test.csv

评价指标:

- **宏精确度 (Macro Precision)**
  $$Macro\_P = \frac{1}{C} \sum_{i=1}^{C} P_i = \frac{1}{C} \sum_{i=1}^{C} \frac{TP_i}{TP_i + FP_i}$$

- **宏召回率 (Macro Recall)**
  $$Macro\_R = \frac{1}{C} \sum_{i=1}^{C} R_i = \frac{1}{C} \sum_{i=1}^{C} \frac{TP_i}{TP_i + FN_i}$$

- **宏 F1 值 (Macro F1-Score)**
  $$Macro\_F1 = \frac{2 \times Macro\_P \times Macro\_R}{Macro\_P + Macro\_R}$$

**公式符号详细说明：**

- $C$ ：类别总数（本任务中为 6 个情绪类别）
- $i$ ：类别索引，取值范围为 1 到 C（即 1 到 6）
- $P_i$ ：第$i$类的精确度（Precision）
- $R_i$ ：第$i$类的召回率（Recall）
- $TP_i$ ：第$i$类的真正例（True Positive）- 正确预测为第$i$类的样本数
- $FP_i$ ：第$i$类的假正例（False Positive）- 错误预测为第$i$类的样本数
- $FN_i$ ：第$i$类的假负例（False Negative）- 实际为第$i$类但预测为其他类的样本数
- $\sum_{i=1}^{C}$ ：对所有类别求和的符号

**举例说明：**
对于"积极"情绪类别（假设为第 1 类）：

- $TP_1$：模型正确识别为"积极"的微博数量
- $FP_1$：模型错误识别为"积极"的微博数量（实际为其他情绪）
- $FN_1$：实际为"积极"但被模型识别为其他情绪的微博数量

宏平均的计算方式是先计算每个类别的指标，然后求平均值，这样可以平等对待每个类别，不受类别样本数量不平衡的影响。

要求同学们以提高在测试集上的效果为目标，自己根据数据特点及需要进行数据预处理以及模型设计。
本任务不对模型的选择和设计进行限制，同时要求训练所用数据不脱离给定的数据集范围不可引入外部语料(例如通过外部语料得到的预训练词向量等)，可以以 CNN、RNN 等模型为基础进行设计
实验报告中需要包含但不限于数据的处理过程介绍(例如分词等)、词向量、模型图、模型中各部分的作用介绍或者使用理由、超参数设置以及训练过程中各指标变化和实验结果分析等。

提示:可使用 python 中的 **gensim** 库在给定语料上进行词向量训练注:为保证公平，在训练过程中同样也不可引入测试数据，测试数据只用于最终评估

## 任务流程

### 数据预处理 (Data Preprocessing)

- 🧹 **文本清洗**
  - 去除 URL、@用户名、特殊符号
  - 处理表情符号和 emoji
  - 统一繁简体转换
- ✂️ **中文分词**
  - 使用 spaCy 或其他先进的分词工具

### 词向量训练 (Word Embedding)

- 📚 **语料准备**
  - 基于训练集和测试集（仅文本内容，不含标签）的清洗后文本进行分词。
- 🎯 **词向量选择/训练**
  - **当前方案: 使用 spaCy 模型生成句子向量**
    - 利用已加载的 spaCy 中文模型 (如 `zh_core_web_sm`)。
    - 对分词后的词语列表（token list），通过 spaCy 的 `Doc` 对象的 `.vector` 属性生成句子级别的向量表示。
    - 这种方法利用模型内部的表示能力，不直接加载外部的预训练词向量文件，符合项目要求。

### 模型设计 (Model Architecture)

- 🏗️ **基础模型选择**
  - **Transformer-based 模型**:
    - BERT (Bidirectional Encoder Representations from Transformers)
    - RoBERTa (A Robustly Optimized BERT Pretraining Approach)
    - ERNIE (Enhanced Representation through kNowledge IntEgration)
    - 其他针对中文优化的预训练模型 (如 MacBERT, Chinese-BERT-wwm)
  - **轻量级 Transformer 模型**:
    - ALBERT (A Lite BERT for Self-supervised Learning of Language Representations)
    - DistilBERT (a distilled version of BERT)
- ⚙️ **网络结构设计**
  - **对于 Transformer-based 模型**:
    - 输入层：使用模型对应的 Tokenizer 对文本进行编码（包括特殊标记如 [CLS], [SEP]）
    - 嵌入层：加载预训练模型的权重
    - Transformer 编码器层：利用预训练模型的强大特征提取能力
    - 池化层 (Pooling)：通常使用 [CLS] token 的输出作为句子表示，或对所有 token 输出进行平均/最大池化
    - 分类层：在池化输出后接一个或多个全连接层 (Dense Layer) + Softmax 进行分类
  - **对于传统深度学习模型**:
    - 嵌入层：加载预训练词向量或随机初始化
    - 特征提取层：CNN 卷积层、RNN/LSTM/GRU 层
    - 池化层 (可选)
    - 分类层：全连接层 + Softmax
- 🎛️ **超参数设置**
  - **通用超参数**:
    - 学习率 (Learning Rate)：通常需要针对不同模型进行调整，预训练模型微调时常使用较小的学习率
    - 批次大小 (Batch Size)
    - Dropout 率
    - 优化器选择 (AdamW, Adam, SGD 等)
  - **Transformer 特定超参数**:
    - 最大序列长度 (Max Sequence Length)
    - 学习率调度器 (Learning Rate Scheduler, e.g., linear warmup with decay)

### 模型训练 (Model Training)

- 🔄 **训练策略**
  - 数据划分：训练集/验证集分割 (通常 80/20 或 90/10)
  - 批次训练和梯度更新
  - 早停机制防止过拟合
- 📈 **训练监控**
  - 记录训练损失和准确率曲线
  - 监控验证集性能变化
  - 保存最佳模型检查点
- ⚖️ **类别平衡处理**
  - 类别权重调整
  - 数据增强技术
  - 采样策略优化

### 5. 模型评估 (Model Evaluation)

- 🎯 **性能评估**
  - 在验证集上计算宏精确度、宏召回率、宏 F1 值
  - 绘制混淆矩阵分析各类别预测效果
  - 错误样本分析和 case study
- 🔧 **模型优化**
  - 超参数调优（网格搜索、贝叶斯优化）
  - 模型集成（投票、平均等）
  - 特征工程改进
