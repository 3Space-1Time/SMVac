import re

with open('paper/main.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Add float package if not exists
if r'\usepackage{float}' not in content:
    content = content.replace(r'\usepackage{graphicx}', r'\usepackage{graphicx}' + '\n' + r'\usepackage{float}')

# Replace figure environments
content = re.sub(r'\\begin\{figure\}\[.*?\]', r'\\begin{figure}[H]', content)
content = re.sub(r'\\begin\{figure\*\}\[.*?\]', r'\\begin{figure*}[H]', content)

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(content)
