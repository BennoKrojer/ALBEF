import os
import shutil

#rename all files in folder from png to jpg

for filename in os.listdir('/home/mila/b/benno.krojer/scratch/svo/images'):
    src = '/home/mila/b/benno.krojer/scratch/svo/images/' + filename
    dst = '/home/mila/b/benno.krojer/scratch/svo/images/' + filename[:-3] + 'jpg'
    os.rename(src, dst)

