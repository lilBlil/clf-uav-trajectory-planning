import subprocess, shutil, os
out = os.path.join(os.getcwd(), '_refs.txt')
subprocess.run(['pdftotext', '-layout', '-f', '5', '-l', '5',
                'CL_paper_IEEE_English_CL_draft.pdf', out])
txt = open(out, encoding='utf-8', errors='ignore').read()
os.remove(out)
# print from REFERENCES onward
i = txt.upper().find('REFERENCES')
print(txt[i:i+3000] if i >= 0 else txt[-3000:])
