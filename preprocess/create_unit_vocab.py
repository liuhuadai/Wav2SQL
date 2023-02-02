

import json


with open('datasets/spider/nl2code-glove,cv_link=true/unit_vocab.json','w') as f:
    vocab = [
    "<UNK>",
    "<BOS>",
    "<EOS>"]
    for i in range(1000):
        vocab.append(str(i))
    json.dump(vocab,f)