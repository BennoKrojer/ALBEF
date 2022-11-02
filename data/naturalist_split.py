import json

data = json.load(open('/home/mila/b/benno.krojer/scratch/neural-naturalist/birds-to-words-v1.0.json', 'r'))
tsv = open('/home/mila/b/benno.krojer/scratch/neural-naturalist/birds-to-words-v1.0.tsv', 'r')

train = []
val = []
test = []

i = 0

for line, core in zip(tsv, data):
    if i == 0:
        i += 1
        continue
    i += 1
    line = line.strip()
    split = line.split('\t')[-3]
    print(split)
    if split == 'train':
        train.append(core)
    elif split == 'val':
        val.append(core)
    elif split == 'test':
        test.append(core)
    else:
        raise ValueError('Unknown split')

with open('/home/mila/b/benno.krojer/scratch/neural-naturalist/train.json', 'w') as f:
    json.dump(train, f, indent=4)

with open('/home/mila/b/benno.krojer/scratch/neural-naturalist/val.json', 'w') as f:
    json.dump(val, f, indent=4)

with open('/home/mila/b/benno.krojer/scratch/neural-naturalist/test.json', 'w') as f:
    json.dump(test, f, indent=4)