from PyPDF2 import PdfReader
from pathlib import Path

# 利用 PyPDF2 读取 PDF 文本
# 利用 AI 清洗文本

def read_pdf_text(pdf_path):
    """
    Extract and return all text from a PDF. Whitespace and empty pages are ignored.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)

def pdf2text_ai(pdf_path):
    """
    Extract text from a PDF and return cleaned text using AI.
    """
    text = read_pdf_text(pdf_path)
    from api import get_ai_response
    prompt = "请清洗一下下面的文本，并把清洗后的结果发给我呢,不需要多余的说明，以2025-08-2-485-001这样的编号开头（经供参考，具体以我下面给的文件为主）:" + text
    cleaned_text = get_ai_response(prompt)
    return cleaned_text

if __name__ == "__main__":
    # 测试代码
    sample_pdf = "data/ori_pdf/1.pdf"  # 替换为你的 PDF 文件路径
    print(pdf2text_ai(sample_pdf)) 
    