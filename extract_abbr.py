import zipfile
import xml.etree.ElementTree as ET
import re
import sys

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path, 'r') as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            # The namespace for w:t (text) in word document xml
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            texts = tree.findall('.//w:t', namespaces)
            return ' '.join([t.text for t in texts if t.text])
    except Exception as e:
        return str(e)

def find_abbreviations(text):
    # Find words with 2 or more uppercase letters, possibly with numbers, often in parentheses
    # like (LLM) or LTT or UEBA
    
    # 1. Look for explicit definitions like "Large Language Model (LLM)"
    # Pattern: words followed by (ABBR)
    # We will try to capture potential definitions.
    pattern = r'\b([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,5})\s+\(([A-Z]{2,})\)'
    matches = re.findall(pattern, text)
    
    abbr_dict = {}
    for definition, abbr in matches:
        abbr_dict[abbr] = definition
        
    # 2. Look for standalone uppercase abbreviations
    standalone = set(re.findall(r'\b[A-Z]{2,}\b', text))
    for abbr in standalone:
        if abbr not in abbr_dict:
            abbr_dict[abbr] = "Unknown / Need Context"
            
    return abbr_dict

if __name__ == "__main__":
    file_path = r"d:\CP_MTH\thesis_final_repport.docx"
    text = extract_text_from_docx(file_path)
    if "No such file" in text or "File is not a zip file" in text:
        print(f"Error reading file: {text}")
        sys.exit(1)
        
    abbrs = find_abbreviations(text)
    
    print("ABBREVIATIONS_START")
    for abbr in sorted(abbrs.keys()):
        print(f"{abbr}|{abbrs[abbr]}")
    print("ABBREVIATIONS_END")
