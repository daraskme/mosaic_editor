content = open('mosaic.py', encoding='utf-8').read()

# auto_detect_skin への参照を探す
idx = content.find('auto_detect_skin')
while idx != -1:
    print(f'pos={idx}:', repr(content[max(0,idx-80):idx+80]))
    print('---')
    idx = content.find('auto_detect_skin', idx+1)
