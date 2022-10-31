import os
import sys

start = int(sys.argv[1])
end = int(sys.argv[2])

for i, line in enumerate(open('search.params/modified_params', 'r')):
    if i >= start and i <= end:
        os.system(line)
