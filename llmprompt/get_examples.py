import json
import random
import sys

fname = 'bigbench-lite/{}.json'.format(sys.argv[1])
tname = 'testsets/{}.json'.format(sys.argv[1])

with open(fname,'r') as f:
    j = json.load(f)
ex = j["examples"]
samp = random.choices(range(len(ex)), k=10)
print(samp)

for i in samp:
    tg = ex[i]["target_scores"]
    print('QUESTION:', ex[i]["input"], end='\\n')
    if tg["a"] == 1:
        print('ANSWER:', 'a', end='|')
    else:
        print('ANSWER:', 'b', end='|')

with open(tname, 'w') as f:
    for i in range(len(ex)):
        if i not in samp:
            f.write(str(json.dumps(ex[i])) + '\n')
