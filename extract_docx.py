import zipfile
import xml.etree.ElementTree as ET

def read_docx(file_path):
    with zipfile.ZipFile(file_path) as docx:
        tree = ET.XML(docx.read('word/document.xml'))
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        # Extract text from w:p (paragraphs) to preserve some line breaks instead of squishing everything together
        paragraphs = []
        for p in tree.findall('.//w:p', namespaces):
            texts = [node.text for node in p.findall('.//w:t', namespaces) if node.text]
            if texts:
                paragraphs.append(''.join(texts))
        return '\n'.join(paragraphs)

text = read_docx(r'd:\CP_MTH\kruthika_10001319_master_thesis_draft.docx')
with open(r'd:\CP_MTH\thesis_text.txt', 'w', encoding='utf-8') as f:
    f.write(text)
