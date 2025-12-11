import erniebot
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 配置 ERNIE Bot
erniebot.api_type = 'aistudio'
# 从环境变量获取 Access Token，如果不存在则报错或使用空字符串
erniebot.access_token = os.getenv("ERNIE_ACCESS_TOKEN")

if not erniebot.access_token:
    print("警告: 未找到 ERNIE_ACCESS_TOKEN 环境变量，请在 .env 文件中配置。")

def generate_web_page(markdown_content):
    """
    使用 ERNIE 模型根据 Markdown 内容生成 HTML 网页代码。
    
    Args:
        markdown_content (str): Markdown 格式的文本内容。
        
    Returns:
        str: 生成的 HTML 代码。
    """
    print("正在请求 ERNIE 生成网页...")
    
    prompt = f"""
    你是一位精通排版和美学的前端设计专家。请将以下 Markdown 内容转换为一个设计精美、极具学术或专业杂志质感的 HTML 网页。

    ### 设计规范：
    1.  **整体风格**：
        - 采用“现代极简”风格，类似 Medium 或专业学术博客的阅读体验。
        - 页面背景色为柔和的灰白色（#f9f9f9），内容区域为白色卡片式设计，带有轻微的阴影（box-shadow）。
        - 内容区域最大宽度限制在 800px-900px，并水平居中，上下内边距充足。

    2.  **排版细节**：
        - **字体**：使用高可读性的无衬线字体（如 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif）。
        - **标题**：颜色深邃（#1a1a1a），字重加粗，上下间距适宜。一级标题居中显示。
        - **正文**：字号 17px，行高 1.8，颜色 #333，确保长时间阅读不疲劳。
        - **段落**：段落之间有舒适的间距。

    3.  **特殊元素**：
        - **表格**：宽度 100%，表头背景色淡雅（#f1f1f1），表头底部有粗边框，行底有细边框，鼠标悬停行高亮。
        - **图片/占位符**：图片居中，圆角 8px，带阴影。对于 `**[图片区域]**` 或 `**[表格内容]**` 这样的占位符，设计一个带有虚线边框、灰色背景和居中文字的占位框，使其在视觉上明显但和谐。
        - **引用**：左侧有 4px 宽的强调色（如 #007bff）边框，背景淡灰，字体样式区别于正文。

    4.  **响应式**：
        - 完美适配移动端，在小屏幕上自动调整内边距和字体大小。

    ### 待转换内容：
    {markdown_content}
    
    ### 输出要求：
    请输出完整的 HTML 代码（包含内联 CSS）。
    请务必使用 ```html 和 ``` 包裹代码。
    """
    
    try:
        # 使用 ERNIE-4.0 或其他可用模型
        response = erniebot.ChatCompletion.create(
            model='ernie-4.0',
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        result = response.get_result()
        
        # 简单的提取逻辑，实际可能需要更复杂的解析
        if "```html" in result:
            html_code = result.split("```html")[1].split("```")[0]
        elif "```" in result:
             html_code = result.split("```")[1].split("```")[0]
        else:
            html_code = result
            
        print("网页生成完成。")
        return html_code
        
    except Exception as e:
        print(f"ERNIE 请求失败: {e}")
        return "<html><body><h1>生成失败</h1></body></html>"

if __name__ == "__main__":
    # 测试代码
    sample_md ="/home/ubuntu/heike/WebBuilder_PaddleOCR_ERNIE/docs/content.md"
    # "# Hello World\nThis is a test."
    print(generate_web_page(sample_md))
