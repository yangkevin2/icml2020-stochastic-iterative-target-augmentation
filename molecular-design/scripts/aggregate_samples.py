from argparse import ArgumentParser
import os

def aggregate(args):
    lines = set()
    for _, dirs, _ in os.walk(args.checkpoint_dir):
        for dirname in dirs:
            if dirname.startswith('epoch'):
                with open(os.path.join(args.checkpoint_dir, dirname, 'train_pairs.csv'), 'r') as rf:
                    for line in rf:
                        lines.add(line.strip())
    with open(args.save_path, 'w') as wf:
        for line in lines:
            wf.write(line + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--save_path', type=str, default=None)
    
    args = parser.parse_args()
    aggregate(args)