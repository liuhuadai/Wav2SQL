from glob import glob
import json
from pathlib import Path
import os, numpy as np
import pickle

for section in ['enc','dec']:
    items = []
    with open(f'datasets/spider/nl2code-glove,cv_link=true/{section}/train.jsonl') as f:
        for line in f:
            items.append(json.loads(line))
    with open(f'datasets/spider/nl2code-glove,cv_link=true/{section}/val.jsonl') as f:
        for line in f:
            items.append(json.loads(line))
    for split in ['train','dev']:
        audio = np.load(os.path.join('spider-dg-real/feature', split+'_0_1.npy'))
        # audio = np.load(f'spider/hubert/{section}_0_1.npy')
        with open(os.path.join('spider-dg-real/feature', split+'_0_1.len')) as f:
            lengths = f.readlines()
        sum = 0 
        questions = []
        with open(f'datasets/spider/nl2code-glove,cv_link=true/{section}/{split}_dg.jsonl','w') as f_w:
            paths = sorted(glob(f'spider-dg-real/{split}/*.wav'), key=lambda x:int(Path(x).stem.split('_')[0][3:]))
            for i, path in enumerate(paths):
                id = int(Path(path).stem.split('_')[0][3:])
                schema = items[id-1]
                schema = json.dumps(schema)
                if section == 'enc':
                    length = int(lengths[i].strip())
                    question = audio[sum:sum+length]
                    sum+=length
                    questions.append(question)
                
                print(schema, file=f_w)
        if section == 'enc':
            pickle.dump(questions, open(f'spider-dg-real/processed_feature/{split}.bin','wb'))




