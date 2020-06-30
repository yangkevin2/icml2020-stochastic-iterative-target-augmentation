import os

os.makedirs('data/1m_6ex_karel/chunks', exist_ok=True)

with open('data/1m_6ex_karel/train.json', 'r') as rf:
    count = 0
    start_index = count
    chunk = []
    for line in rf:
        count += 1
        chunk.append(line)
        if len(chunk) == 10000:
            with open('data/1m_6ex_karel/chunks/train' + str(start_index) + '_' + str(count) + '.json', 'w') as wf:
                for line in chunk:
                    wf.write(line)
            chunk = []
            start_index = count
    if len(chunk) > 0:
        with open('data/1m_6ex_karel/chunks/train' + str(start_index) + '_' + str(count) + '.json', 'w') as wf:
            for line in chunk:
                wf.write(line)
        chunk = []
        start_index = count