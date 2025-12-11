import os
import argparse
from ocr_processor import process_pdf
from ernie_generator import generate_web_page
from experiment_manager import ExperimentManager

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="PaddleOCR + ERNIE Web Builder")
    parser.add_argument('--model_version', type=str, default='baseline', help='模型版本标识 (例如: baseline, v1, v2)')
    args = parser.parse_args()
    
    model_version = args.model_version

    # 路径配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    
    # 初始化实验管理器
    experiment_manager = ExperimentManager(results_dir)
    
    # 查找 data 目录下的 PDF 文件
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"在 {data_dir} 中未找到 PDF 文件。请放入 PDF 文件后重试。")
        return

    # 处理第一个 PDF 文件 (示例)
    pdf_file = pdf_files[0]
    pdf_name = os.path.splitext(pdf_file)[0]
    pdf_path = os.path.join(data_dir, pdf_file)
    
    print(f"开始处理任务: {pdf_file} (Model: {model_version})")
    
    # 为该样本和模型版本创建独立的输出目录
    # 结构: results/sample_name/model_version/
    sample_model_dir = os.path.join(results_dir, pdf_name, model_version)
    
    # 1. OCR 处理结果目录
    ocr_output_dir = os.path.join(sample_model_dir, 'step1_ocr')
    if not os.path.exists(ocr_output_dir):
        os.makedirs(ocr_output_dir)
        
    # 2. 网页生成结果目录
    web_output_dir = os.path.join(sample_model_dir, 'step2_web')
    if not os.path.exists(web_output_dir):
        os.makedirs(web_output_dir)
        
    # 3. 图表保存目录 (在样本根目录下，用于跨模型对比)
    charts_dir = os.path.join(results_dir, pdf_name, 'comparison_charts')
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    
    # --- Step 1: OCR 处理 ---
    # 传入输出目录以保存图片等，并传入模型版本
    markdown_content, metrics = process_pdf(pdf_path, ocr_output_dir, model_version)
    
    # 保存中间结果 (Markdown)
    with open(os.path.join(ocr_output_dir, 'content.md'), 'w', encoding='utf-8') as f:
        f.write(markdown_content)
        
    # --- Step 2: 记录实验数据 ---
    experiment_manager.log_experiment(pdf_name, model_version, metrics)
    
    # --- Step 3: 绘制对比图表 ---
    # 只有当有多个模型的数据时，对比图才有意义，但每次运行都更新也没问题
    experiment_manager.plot_charts(pdf_name, charts_dir)
        
    # --- Step 4: 网页生成 ---
    html_content = generate_web_page(markdown_content)
    
    # 保存结果
    output_html_path = os.path.join(web_output_dir, 'index.html')
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"任务完成！")
    print(f"OCR 结果: {ocr_output_dir}")
    print(f"网页结果: {output_html_path}")
    print(f"对比图表: {charts_dir}")

if __name__ == "__main__":
    main()
