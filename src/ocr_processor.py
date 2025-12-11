import os
import fitz  # PyMuPDF
import cv2
import numpy as np
import gc
from paddleocr import PPStructureV3

def process_pdf(pdf_path):
    """
    使用 PaddleOCR-VL (PP-Structure) 处理 PDF 文件并转换为 Markdown 文本。
    
    Args:
        pdf_path (str): PDF 文件的路径。
        
    Returns:
        str: 转换后的 Markdown 内容。
    """
    print(f"正在处理 PDF: {pdf_path} ...")
    
    # 初始化 PP-Structure 引擎
    # use_table_recognition=False 表示不进行表格结构识别，仅做版面分析和文本识别，速度更快
    # 如果需要表格识别，设置 use_table_recognition=True
    engine = PPStructureV3(use_table_recognition=False, lang='ch')
    
    markdown_lines = []
    
    try:
        # 打开 PDF
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            print(f"正在处理第 {page_num + 1} 页...")
            
            # 将 PDF 页面转换为图像
            # 降低 DPI 以减少内存占用 (300 -> 150)
            pix = page.get_pixmap(dpi=150)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # 如果是 RGBA，转为 RGB
            if pix.n == 4:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1: # Grayscale
                img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
                
            # 运行 PP-Structure
            # engine.predict 返回一个列表，我们需要获取第一个结果（因为我们一次只处理一张图片）
            predict_result = engine.predict(img_data)[0]
            
            # 解析结果并转换为 Markdown
            # PPStructureV3 返回的结果中包含 parsing_res_list
            parsing_res_list = predict_result['parsing_res_list']
            
            for region in parsing_res_list:
                res_type = region.label.lower() # 确保是小写
                
                if res_type == 'title':
                    # 提取标题文本
                    text = region.content
                    markdown_lines.append(f"## {text}\n")
                    
                elif res_type == 'text':
                    # 提取正文文本
                    text = region.content
                    markdown_lines.append(f"{text}\n")
                    
                elif res_type == 'table':
                    # 如果开启了表格识别，content 可能是 html 或者 text
                    # 这里简单处理，直接使用 content
                    text = region.content
                    markdown_lines.append(f"**[表格内容]** {text}\n")
                        
                elif res_type == 'figure' or res_type == 'image':
                    markdown_lines.append(f"**[图片区域]**\n")
                    
                elif res_type == 'header':
                     # 页眉通常忽略，或者作为引用
                     pass
                     
                elif res_type == 'footer':
                     # 页脚通常忽略
                     pass
                     
            markdown_lines.append("\n---\n") # 分页符

            # 手动释放内存
            del pix
            del img_data
            del predict_result
            gc.collect()

        doc.close()
        
    except Exception as e:
        print(f"处理 PDF 时发生错误: {e}")
        return f"# 错误\n处理文件 {pdf_path} 时出错: {str(e)}"
    
    markdown_content = "\n".join(markdown_lines)
    print("PDF 处理完成。")
    return markdown_content

if __name__ == "__main__":
    # 测试代码
    test_pdf = "/home/ubuntu/heike/WebBuilder_PaddleOCR_ERNIE/data/sample.pdf"
    if os.path.exists(test_pdf):
        print(process_pdf(test_pdf))
    else:
        print(f"文件不存在: {test_pdf}")
