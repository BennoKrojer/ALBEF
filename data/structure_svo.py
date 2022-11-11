import json
import os
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

#load csv
df = pd.read_csv('/home/mila/b/benno.krojer/scratch/svo/svo_probes.csv')
d = []

print(df.columns)

#iterate over rows

for index, row in tqdm(df.iterrows()):
    sentence = row['sentence']
    pos_url = row['pos_url']
    neg_url = row['neg_url']
    pos_image_id = row['pos_image_id']
    neg_image_id = row['neg_image_id']
    
    pos_url = pos_url.replace('/', '_')
    neg_url = neg_url.replace('/', '_')
    extension_pos = pos_url.split('.')[-1]
    extension_neg = neg_url.split('.')[-1]
    if extension_pos != 'jpg' or extension_neg != 'jpg' or extension_neg == 'png' or extension_pos == 'png':
        continue
    try:
        # copy images with shutil
        shutil.copyfile(f'/home/mila/b/benno.krojer/scratch/svo/old-images/{pos_url}', f'/home/mila/b/benno.krojer/scratch/svo/images/{pos_image_id}.{extension_pos}')
        shutil.copyfile(f'/home/mila/b/benno.krojer/scratch/svo/old-images/{neg_url}', f'/home/mila/b/benno.krojer/scratch/svo/images/{neg_image_id}.{extension_neg}')
    except:
        # show error
        print(f'error with {pos_url} and {neg_url}')
        continue
    d.append({
        'pos_id': f'{pos_image_id}.{extension_pos}',
        'neg_id': f'{neg_image_id}.{extension_neg}',
        'sentence': sentence,
    })

# shuffle data
np.random.shuffle(d)
#split data
train = d[:int(len(d)*0.8)]
test = d[int(len(d)*0.8):]

# write data
with open('/home/mila/b/benno.krojer/scratch/svo/train.json', 'w') as f:
    json.dump(train, f, indent=4)

with open('/home/mila/b/benno.krojer/scratch/svo/val.json', 'w') as f:
    json.dump(val, f, indent=4)

with open('/home/mila/b/benno.krojer/scratch/svo/test.json', 'w') as f:
    json.dump(test, f, indent=4)