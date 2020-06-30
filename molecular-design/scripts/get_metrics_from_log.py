import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str)
parser.add_argument('--metric', type=str)
args = parser.parse_args()

with open(args.log) as f:
    for line in f:
        if args.metric in line:
            # line = line.strip().split('=')[1].split('+')[0].strip()
            line = line.strip().split(' ')[0]
            print(line)