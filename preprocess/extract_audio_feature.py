import os
import json
import speechpy
import scipy.io.wavfile as wav
import numpy as np

for split in ['train','val']:
    i = 0
    # output_path = f'/home/huangrongjie1/projects/rat-sql/datasets/spider/nl2code-glove,cv_link=true/enc/{split}.jsonl'
    # f = open(output_path,'w')
    for line in open(f'/home/huangrongjie1/projects/rat-sql/datasets/spider/nl2code-glove,cv_link=true/{split}.jsonl'):
        i+=1
        input = json.loads(line)
        question = input['raw_question']
        if split == 'val':
            path = f'/home/huangrongjie1/projects/FastSpeech2/output/result/val/{i}.wav'
        else:
            path = f'/home/huangrongjie1/projects/FastSpeech2/output/result/train/{i}.wav'
        fs, signal = wav.read(path)
        # signal = signal[:, 0]
        logenergy = speechpy.feature.lmfe(
            signal, sampling_frequency=22050, frame_length=0.025, frame_stride=0.01, num_filters=80)
        cmvn_feature = speechpy.processing.cmvn(
            logenergy, variance_normalization=True)
        np.save(f'/home/huangrongjie1/projects/rat-sql/datasets/spider/nl2code-glove,cv_link=true/enc/wav_features/{split}/{i}.npy',cmvn_feature)
        # input['audio'] = cmvn_feature
        # output = json.dumps(input)
        # f.write(output)
    # f.close()
