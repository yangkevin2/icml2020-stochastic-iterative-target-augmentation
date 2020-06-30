import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import rdkit.Chem.QED as QED
from .sascorer import calculateScore
from .drd2_scorer import get_score as get_drd2_score
import networkx as nx

def similarity(a, b, chiral=True):
    if a is None or b is None: return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=chiral)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=chiral)
    return DataStructs.TanimotoSimilarity(fp1, fp2) 

def drd2(s):
    if s is None: return 0.0
    return get_drd2_score(s)

def mol_weight(s):
    if s is None: return 0.0
    mol = Chem.MolFromSmiles(s)
    return Descriptors.ExactMolWt(mol)

def qed(s):
    if s is None: return 0.0
    mol = Chem.MolFromSmiles(s)
    try:
        qed_score = QED.qed(mol)
    except:
        qed_score = 0
    return qed_score

def sascore(s):
    if s is None: return 0.0
    mol = Chem.MolFromSmiles(s)
    return -calculateScore(mol)

def logp(s):
    if s is None: return -100.0
    mol = Chem.MolFromSmiles(s)
    return Descriptors.MolLogP(mol)

def penalized_logp(s):
    if s is None: return -100.0
    mol = Chem.MolFromSmiles(s)

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle

def smiles2D(s):
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol)

if __name__ == "__main__":
    print(round(penalized_logp('ClC1=CC=C2C(C=C(C(C)=O)C(C(NC3=CC(NC(NC4=CC(C5=C(C)C=CC=C5)=CC=C4)=O)=CC=C3)=O)=C2)=C1'), 2), 5.30)
