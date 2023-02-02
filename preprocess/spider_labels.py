import os
from num2words import num2words
import json

# print(num2words(10244,to='year'))
for split in ['train','val']:
    with open(f'datasets/spider/nl2code-glove,cv_link=true/enc/{split}.jsonl') as f, open(
        f'datasets/spider/nl2code-glove,cv_link=true/enc/{split}.ltr', "w"
    ) as ltr_out, open(
        f'datasets/spider/nl2code-glove,cv_link=true/enc/{split}.wrd', "w"
    ) as wrd_out:
        for line in f:
            line = json.loads(line)
            question = line['raw_question']
            question = list([val for val in question if val.isalpha() or val == ' ' or val.isnumeric() ]) 
            question = ''.join(question)
            q_tokens = question.split()
            result = []
            for val in q_tokens:
                if val.isnumeric():
                    vals = num2words(val,to='year').replace('-',' ').replace(',','').upper().split()
                    for i in vals:
                        result.append(i)
                if val.isalpha():
                    result.append(val.upper())
            result = " ".join(result)
            print(result, file=wrd_out)
            print(
                " ".join(list(result.replace(" ", "|"))) + " |",
                file=ltr_out,
            )
            # q_tokens = list([num2words(val,to='year').replace('-',' ').replace(',','') for val in q_tokens 
            # if val.isnumeric() ]) 
            # getVals = list([val for val in q_tokens 
            # if val.isalpha() ]) 
            # result = " ".join(getVals)
            
