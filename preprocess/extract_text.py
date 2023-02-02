

import json
i = 1
for split in ['train','val']:
    with open(f'datasets/spider/nl2code-glove,cv_link=true/enc/{split}.jsonl') as f, open(f'datasets/spider/nl2code-glove,cv_link=true/enc/{split}_reduced.jsonl','w') as f_out:
        for line in f:
            line = json.loads(line)
            units = line['question']
            out = [u for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
            line['question'] = out
            # db_id = line['db_id']
            # out = str(i)+'\t'+text+'\t'+db_id
            # print(out,file=f_out)
            line = json.dumps(line)
            print(line, file=f_out)
            i+=1
