# PaddleOCR-VL 微调实验与论文数据生成指南

本文档详细说明了如何利用本项目的新增功能，自动化地进行模型微调实验记录、数据收集与图表生成，从而为您的学术论文提供强有力的定性与定量数据支持。

## 一、 实验设计理念 (Design Philosophy)

为了完美展示微调与优化的过程，我们将项目从一个简单的“转换工具”升级为了一个“实验平台”。核心设计理念如下：

1.  **结构化存储**: 每次实验（不同模型版本）的数据严格隔离，便于横向对比。
2.  **自动化记录**: 自动记录关键性能指标（置信度、耗时、检出区域数），无需人工统计。
3.  **可视化输出**: 自动生成对比图表，直接满足论文插图需求。

## 二、 目录结构说明 (Directory Structure)

运行实验后，`results/` 目录将自动生成如下结构。这就是您的“实验数据库”。

```text
results/
├── experiment_log.csv          <-- [核心数据] 自动追加记录所有实验的量化指标（时间、置信度等）
└── sample/                     <-- 针对 "sample.pdf" 的专属实验记录
    ├── comparison_charts/      <-- [论文插图] 自动生成的对比图表
    │   ├── confidence_comparison.png  <-- 不同模型版本的置信度对比柱状图
    │   └── time_comparison.png        <-- 不同模型版本的耗时对比柱状图
    │
    ├── baseline/               <-- [实验 1] 基线模型结果
    │   ├── step1_ocr/          <-- OCR 原始输出
    │   │   ├── content.md      <-- 识别出的文本
    │   │   └── images/         <-- 提取出的图片与表格（用于定性分析）
    │   └── step2_web/          <-- 网页生成结果
    │
    ├── finetuned_v1/           <-- [实验 2] 微调版本 V1 结果
    │   ├── step1_ocr/
    │   └── step2_web/
    │
    └── finetuned_v2/           <-- [实验 3] 微调版本 V2 结果...
```

## 三、 核心功能模块 (Core Modules)

我们对三个核心文件进行了升级以支持此流程：

1.  **`src/experiment_manager.py` (新增)**
    *   **功能**: 实验数据的“管家”。
    *   **作用**: 负责将每次运行的指标写入 CSV 文件，并利用 Pandas 和 Seaborn 绘制精美的对比图表。

2.  **`src/ocr_processor.py` (增强)**
    *   **功能**: OCR 核心处理引擎。
    *   **升级**: 新增了指标计算功能，现在返回 `(markdown_content, metrics)` 元组。
    *   **指标**:
        *   `avg_confidence`: 平均置信度（衡量模型“确信”程度）。
        *   `processing_time`: 处理耗时（衡量模型速度）。
        *   `regions_detected`: 检出区域数量（衡量检测召回率）。

3.  **`src/main.py` (升级)**
    *   **功能**: 主程序入口。
    *   **升级**: 增加了命令行参数 `--model_version`，用于区分不同的实验批次。

## 四、 实验操作流程 (Workflow)

请按照以下步骤模拟或执行您的微调实验。

### 第一步：运行基线模型 (Baseline)
假设这是未微调的原始模型（或使用官方预训练模型）。

```bash
python src/main.py --model_version baseline
```
*   **产出**: 建立 `results/sample/baseline` 目录，生成初始日志。

### 第二步：运行微调模型 V1 (Finetuned V1)
假设您优化了数据增强策略（Data Augmentation）后得到的模型。

```bash
python src/main.py --model_version finetuned_v1
```
*   **产出**: 建立 `results/sample/finetuned_v1` 目录。
*   **变化**: `comparison_charts/` 中的图表会自动更新，展示 Baseline vs V1 的对比。

### 第三步：运行微调模型 V2 (Finetuned V2)
假设您进一步优化了模型架构（Model Architecture）或使用了难例挖掘（Hard Negative Mining）。

```bash
python src/main.py --model_version finetuned_v2
```
*   **产出**: 建立 `results/sample/finetuned_v2` 目录。
*   **变化**: 图表现在包含三组柱状图，清晰展示性能随版本的变化趋势。

## 五、 素材 (For Paper)

### 1. 定性分析 (Qualitative Analysis)
*   **素材来源**: `results/sample/*/step1_ocr/images/`
*   **写作方法**: 挑选一张具有挑战性的表格或图片（例如 `table_0_1.jpg`），分别从 `baseline` 和 `finetuned_v2` 文件夹中取出。
*   **展示**: 将两张图片并排放在论文中。
*   **描述**: “如图 X 所示，Baseline 模型（左）在处理复杂表格线时出现了断裂，而微调后的 V2 模型（右）成功还原了表格结构...”

### 2. 定量分析 (Quantitative Analysis)
*   **素材来源**: `results/sample/comparison_charts/` 和 `results/experiment_log.csv`
*   **写作方法**: 直接插入生成的 `confidence_comparison.png`。
*   **描述**: “实验结果表明（见图 Y），随着微调策略的引入，模型的平均置信度从 Baseline 的 0.85 提升至 V2 的 0.96，显著提高了识别的可靠性。同时，推理时间保持在可接受范围内...”


