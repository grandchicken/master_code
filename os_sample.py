import os

for root,dirs,files in os.walk(r'D:\njunlp\sample_program\text_classification'):
    for name in files:
        print(os.path.join(root,name))
    # for name in dirs:
    #     print(os.path.join(root,name))ee