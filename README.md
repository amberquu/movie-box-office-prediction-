# Movie Box Office Prediction

**[English](#-english) | [中文](#-中文)**

---

## English

### Overview

A machine learning project that investigates how budget, popularity and audience ratings influence a film's box office revenue. The project covers the full data science workflow: raw data ingestion, cleaning, exploratory analysis, feature engineering, model training and interactive business dashboards — achieving 80% prediction accuracy.

### Features

- **End-to-end Python Pipeline** — From raw CSV to trained model in a single reproducible workflow
- **Exploratory Data Analysis** — Correlation analysis and visualisations of budget, popularity, vote average and revenue relationships
- **Feature Engineering** — Derived features including budget tiers, popularity bins and release-season flags
- **Multiple Models** — Decision Tree and Logistic Regression compared side by side
- **80% Prediction Accuracy** — Achieved on held-out test set after hyperparameter tuning
- **Power BI Dashboard** — Interactive visuals for presenting findings to non-technical stakeholders

### Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Python, pandas |
| Modelling | scikit-learn (Decision Tree, Logistic Regression) |
| Visualisation (code) | matplotlib, seaborn |
| Business Dashboard | Power BI |

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | **80%** | 0.81 | 0.79 | 0.80 |
| Logistic Regression | 74% | 0.75 | 0.73 | 0.74 |

Decision Tree outperformed Logistic Regression, likely due to non-linear interactions between budget and popularity tiers.

### Key Findings

- Budget is the single strongest predictor of box office success
- Films with above-median popularity scores perform disproportionately better, suggesting marketing and social buzz matter as much as production spend
- Release season (summer / holiday) adds a meaningful signal on top of budget alone
- Vote average alone is a weak predictor — critical reception does not reliably translate to commercial performance

### Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/movie-prediction.git
cd movie-prediction

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/pipeline.py

# Or step through individual stages
python src/preprocess.py      # Data cleaning
python src/eda.py             # Exploratory analysis & plots
python src/train.py           # Model training & evaluation
```

### Project Structure

```
movie-prediction/
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned & feature-engineered data
├── src/
│   ├── preprocess.py         # Data cleaning & feature engineering
│   ├── eda.py                # EDA visualisations
│   ├── train.py              # Model training & evaluation
│   └── pipeline.py           # End-to-end runner
├── notebooks/
│   └── exploration.ipynb     # Interactive analysis notebook
├── dashboards/
│   └── box_office.pbix       # Power BI dashboard file
├── outputs/
│   ├── figures/              # Saved plots
│   └── models/               # Serialised model files
└── requirements.txt
```

### Pipeline Overview

```
Raw Data (CSV)
     │
     ▼
Data Cleaning
  - Handle missing values
  - Remove duplicates & outliers
  - Standardise formats
     │
     ▼
Feature Engineering
  - Budget tiers
  - Popularity bins
  - Release season flags
     │
     ▼
Model Training
  - Train/test split (80/20)
  - Decision Tree
  - Logistic Regression
  - Cross-validation
     │
     ▼
Evaluation & Reporting
  - Accuracy, Precision, Recall, F1
  - Power BI dashboard
```

---

## 中文

### 项目简介

一个研究电影制作预算、热度和观众评分如何影响票房收入的机器学习项目。完整覆盖数据科学全流程：原始数据摄取、清洗、探索性分析、特征工程、模型训练，最终用 Power BI 制作业务看板——预测准确率达到 80%。

### 功能亮点

- **端到端 Python 管道** — 从原始 CSV 到训练好的模型，一套脚本跑完全程
- **探索性数据分析** — 对预算、热度、评分和票房之间的相关性进行可视化分析
- **特征工程** — 衍生特征包括预算分层、热度分箱和上映档期标记
- **多模型对比** — 决策树与逻辑回归并排比较
- **80% 预测准确率** — 在留出测试集上经超参数调优后达成
- **Power BI 看板** — 面向非技术受众的交互式可视化报告

### 技术栈

| 层级 | 工具 |
|------|------|
| 数据处理 | Python、pandas |
| 模型训练 | scikit-learn（决策树、逻辑回归） |
| 代码可视化 | matplotlib、seaborn |
| 业务看板 | Power BI |

### 模型表现

| 模型 | 准确率 | 精确率 | 召回率 | F1 分数 |
|------|--------|--------|--------|---------|
| 决策树 | **80%** | 0.81 | 0.79 | 0.80 |
| 逻辑回归 | 74% | 0.75 | 0.73 | 0.74 |

决策树表现更好，可能是因为预算和热度分层之间存在非线性交互关系，决策树能更好地捕捉这类模式。

### 核心发现

- 预算是预测票房最强的单一特征
- 热度高于中位数的影片票房表现远超比例，说明营销和社交讨论度和制作投入同样重要
- 上映档期（暑期档 / 节假日档）在预算之外提供了额外的预测信号
- 评分单独来看是弱预测变量——口碑好并不等于商业成功

### 快速上手

```bash
# 克隆仓库
git clone https://github.com/<your-username>/movie-prediction.git
cd movie-prediction

# 安装依赖
pip install -r requirements.txt

# 运行完整管道
python src/pipeline.py

# 或分步运行
python src/preprocess.py      # 数据清洗
python src/eda.py             # 探索性分析与图表
python src/train.py           # 模型训练与评估
```

### 目录结构

```
movie-prediction/
├── data/
│   ├── raw/                  # 原始数据集
│   └── processed/            # 清洗后的特征数据
├── src/
│   ├── preprocess.py         # 数据清洗与特征工程
│   ├── eda.py                # EDA 可视化
│   ├── train.py              # 模型训练与评估
│   └── pipeline.py           # 全流程入口
├── notebooks/
│   └── exploration.ipynb     # 交互式分析笔记本
├── dashboards/
│   └── box_office.pbix       # Power BI 看板文件
├── outputs/
│   ├── figures/              # 保存的图表
│   └── models/               # 序列化模型文件
└── requirements.txt
```

### 管道流程

```
原始数据（CSV）
     │
     ▼
数据清洗
  - 处理缺失值
  - 去重 & 去异常值
  - 格式标准化
     │
     ▼
特征工程
  - 预算分层
  - 热度分箱
  - 上映档期标记
     │
     ▼
模型训练
  - 训练/测试集划分（80/20）
  - 决策树
  - 逻辑回归
  - 交叉验证
     │
     ▼
评估与报告
  - 准确率、精确率、召回率、F1
  - Power BI 看板
```
