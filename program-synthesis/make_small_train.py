import os

with open('data/1m_6ex_karel/train.json', 'r') as rf:
    count = 0
    if not os.path.exists('data/1m_6ex_karel/chunks/1m_10k'):
        os.makedirs('data/1m_6ex_karel/chunks/1m_10k')
    for line in rf:
        if count % 10000 == 0:
            wf = open('data/1m_6ex_karel/chunks/1m_10k/' + str(count) + '.json', 'w')
        wf.write(line)
        count += 1
        if count % 10000 == 0:
            wf.close()
    wf.close()