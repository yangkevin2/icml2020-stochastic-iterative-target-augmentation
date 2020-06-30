from argparse import ArgumentParser
import random

def sample(args):
    lines = []
    with open(args.load_path, 'r') as rf:
        for line in rf:
            lines.append(line.strip())
    random.shuffle(lines)
    with open(args.save_path, 'w') as wf:
        for line in lines[:args.sample_size]:
            wf.write(line + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--sample_size', type=int)

    args = parser.parse_args()
    sample(args)