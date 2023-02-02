import json
from glob import glob
from pathlib import Path
for split in ['train','dev']:
    lines = []
    for line, path in zip(open(f'datasets/spider/nl2code-glove,cv_link=true/enc/{split}_dg.jsonl'),\
        sorted(glob(f'spider-dg-real/{split}/*.wav'), key=lambda x:int(Path(x).stem.split('_')[0][3:])) ):
        line = json.loads(line)
        spk_id = int(Path(path).stem.split('_')[1][3:])
        if split=='train':
            if spk_id<7:
                line['spk_id'] = spk_id-2
            else:
                line['spk_id'] = spk_id-4
        else:
            line['spk_id'] = spk_id+2
                
        line = json.dumps(line)
        lines.append(line)
    with open(f'datasets/spider/nl2code-glove,cv_link=true/enc/{split}_dg.jsonl','w') as f_w:
        for line in lines:
            print(line, file=f_w)

