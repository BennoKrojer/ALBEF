import json
import os
from tqdm import tqdm
import requests

data = json.load(open('/home/mila/b/benno.krojer/scratch/neural-naturalist/birds-to-words-v1.0.json', 'r'))
ids = set()
for d in data:
    img1 = d['img1_id']
    img2 = d['img2_id']
    ids.add(img1)
    ids.add(img2)

file_extensions = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'png', 'PNG']

for id in tqdm(ids):
    for ext in file_extensions:
        if os.path.exists(f'/home/mila/b/benno.krojer/scratch/neural-naturalist/images/{id}.{ext}'):        
            break
        url = f'https://inaturalist-open-data.s3.amazonaws.com/photos/{id}/large.{ext}'
        r = requests.get(url, allow_redirects=True)
        if r.status_code == 200:
            os.system(f'wget {url} -O /home/mila/b/benno.krojer/scratch/neural-naturalist/images/{id}.{ext}')
            break
