

import json


for split in ['train','val']:
    with open(f'spider/wavs/{split}.txt') as f:
        lines = f.readlines()
    with open(f'datasets/spider/nl2code-glove,cv_link=true/enc/{split}.jsonl') as f_r, \
        open(f'datasets/spider/{split}.jsonl','w') as f_w:
        for i, line in enumerate(f_r):
            line = json.loads(line)
            units = [int(unit) for unit in lines[i].split('|')[1].split()]
            line['question'] = units
            line = json.dumps(line)
            print(line, file=f_w)
