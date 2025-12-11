import os
from ocr_processor import process_pdf
from ernie_generator import generate_web_page

def main():
    # 路径配置
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    docs_dir = os.path.join(base_dir, 'docs')
    
    # 确保输出目录存在
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        
    # 查找 data 目录下的 PDF 文件
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"在 {data_dir} 中未找到 PDF 文件。请放入 PDF 文件后重试。")
        return

    # 处理第一个 PDF 文件 (示例)
    pdf_file = pdf_files[0]
    pdf_path = os.path.join(data_dir, pdf_file)
    
    print(f"开始处理任务: {pdf_file}")
    
    # 1. PDF 转 Markdown
    markdown_content = process_pdf(pdf_path)
    
    # 保存中间结果 (可选)
    with open(os.path.join(docs_dir, 'content.md'), 'w', encoding='utf-8') as f:
        f.write(markdown_content)
        
    # 2. Markdown 转 HTML (通过 ERNIE)
    html_content = generate_web_page(markdown_content)
    
    # 3. 保存结果
    output_html_path = os.path.join(docs_dir, 'index.html')
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"任务完成！网页已生成: {output_html_path}")
    print("请将 docs/ 目录推送到 GitHub 并开启 GitHub Pages 功能。")

if __name__ == "__main__":
    main()
