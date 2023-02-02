import json
import os
import numpy as np
from indexed_datasets import IndexedDatasetBuilder
import attr
binary_data_dir = '/home1/liuhuadai/Transfer/huangrongjie1/projects/Speech2SQL-v3/datasets/spider/nl2code-glove,cv_link=true/enc'

@attr.s
class NL2CodeDecoderPreprocItem:
    tree = attr.ib()
    orig_code = attr.ib()

def process():
    os.makedirs(binary_data_dir, exist_ok=True)
    process_data('val')
    # process_data('test')
    process_data('train')
def process_data(prefix):
    data_dir = binary_data_dir
    builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
    audio = np.load(os.path.join(data_dir, prefix+'_0_1.npy'))
    with open(os.path.join(data_dir, prefix+'_0_1.len')) as f:
        lengths = f.readlines()
    feature_lengths = []
    sum = 0 
    with open(f'datasets/spider/nl2code-glove,cv_link=true/dec/{prefix}.jsonl') as f:
        dec_items = []
        for line in f:
            line = json.loads(line)
            dec_items.append(line)
    with open(f'datasets/spider/nl2code-glove,cv_link=true/enc/{prefix}.jsonl') as f:
        for i, line in enumerate(f):
            length = int(lengths[i].strip())
            input = json.loads(line)
            input['feature'] = audio[sum:sum+length][:500]
            input['len'] = length
            input['sql'] = dec_items[i]
            feature_lengths.append(length)
            builder.add_item(input)
            sum+=length
    builder.finalize()
    np.save(f'{data_dir}/{prefix}_lengths.npy', feature_lengths)

if __name__ == '__main__':
    process()