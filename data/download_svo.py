import json
import os
from tqdm import tqdm
import requests

urls = open('/home/mila/b/benno.krojer/scratch/svo/image_urls.txt', 'r')
for url in tqdm(urls):
    url = url.strip()
    img_id = url.replace('/', '_')
    print("\n\n\n")
    print(url)
    if os.path.exists(f'/home/mila/b/benno.krojer/scratch/svo/images/{img_id}'):
        print("exists")
        continue
    os.system(f'wget {url} -O /home/mila/b/benno.krojer/scratch/svo/images/{img_id}')
    # print(f'wget {url} -O /home/mila/b/benno.krojer/scratch/svo/images/{img_id}')