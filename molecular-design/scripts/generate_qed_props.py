import os

from rdkit import Chem
import rdkit.Chem.QED as QED

os.makedirs('data/qed_prop', exist_ok=True)
with open('data/qed/train_pairs.txt', 'r') as rf, open('data/qed_prop/train_pairs.txt', 'w') as wf:
    for line in rf:
        line = line.strip().split(' ')
        src, tgt = line[0], line[1]
        src_mol, tgt_mol = Chem.MolFromSmiles(src), Chem.MolFromSmiles(tgt)
        src_prop, tgt_prop = QED.qed(src_mol), QED.qed(tgt_mol)
        wf.write(src + ' ' + str(src_prop) + ' ' + tgt + ' ' + str(tgt_prop) + '\n')

with open('data/qed/valid.txt', 'r') as rf, open('data/qed_prop/valid.txt', 'w') as wf:
    for line in rf:
        line = line.strip()
        src_mol = Chem.MolFromSmiles(line)
        src_prop = QED.qed(src_mol)
        wf.write(line + ' '  + str(src_prop) + '\n')

with open('data/qed/test.txt', 'r') as rf, open('data/qed_prop/test.txt', 'w') as wf:
    for line in rf:
        line = line.strip()
        src_mol = Chem.MolFromSmiles(line)
        src_prop = QED.qed(src_mol)
        wf.write(line + ' '  + str(src_prop) + '\n')