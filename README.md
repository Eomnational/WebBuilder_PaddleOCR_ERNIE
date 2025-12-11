# Web Builder: Build a Web Page with PaddleOCR & ERNIE

## 项目简介
本项目旨在完成黑客马拉松的热身任务：使用 PaddleOCR-VL 从 PDF 中提取文本和布局，将其转换为 Markdown，然后利用 ERNIE 模型生成网页，并最终部署到 GitHub Pages。

## 目录结构
- `data/`: 存放输入的 PDF 文件。
- `docs/`: 存放生成的网页文件 (HTML/CSS/JS)，用于 GitHub Pages 部署。
- `src/`: 源代码目录。
  - `ocr_processor.py`: 处理 PDF，提取内容并转换为 Markdown。
  - `ernie_generator.py`: 调用 ERNIE API 生成网页代码。
  - `main.py`: 主程序入口。
- `requirements.txt`: 项目依赖。

## 使用说明
1. 安装依赖: `pip install -r requirements.txt`
2. 配置 ERNIE API Key (在 `src/ernie_generator.py` 或环境变量中)。
3. 将 PDF 文件放入 `data/` 目录。
4. 运行主程序: `python src/main.py`
5. 生成的网页将位于 `docs/` 目录，推送到 GitHub 仓库即可通过 GitHub Pages 访问。

## 任务流程
1. **PDF 解析**: 使用 PaddleOCR-VL 识别 PDF 内容。
2. **Markdown 转换**: 将识别结果整理为 Markdown 格式。
3. **网页生成**: 将 Markdown 发送给 ERNIE，请求生成 HTML 页面。
4. **部署**: 提交 `docs/` 目录到 GitHub。
