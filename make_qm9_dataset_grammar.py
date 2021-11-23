from __future__ import print_function
import nltk
import pdb
import qm9_grammar
import numpy as np
import h5py
import molecule_vae
import pickle
import random
from rdkit import Chem
random.seed(42)

# opening the dataset with the SMILES and properties
with open('data/QM9_STAR.pkl', 'rb') as data:
    df = pickle.load(data)
list_df = list(df.loc[:, 'SMILES_GDB-17'])

# canonicalizing the molecules
list_df = [Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=True, canonical=True) for mol in list_df]

# choosing random ids for train and test
ids = list(range(len(list_df)))
random.shuffle(ids)

# 4% for testing ~ 5000 molecules 
chunk = int(0.04 * len(list_df))
ids_train = sorted(ids[chunk:])
ids_test = sorted(ids[0:chunk])

L = [list_df[i] for i in ids_train] # smiles for training
L_test = [list_df[i] for i in ids_test] # smiles for test

MAX_LEN=100
NCHARS = len(qm9_grammar.GCFG.productions())

def to_one_hot(smiles):
    """ Encode a list of smiles strings to one-hot vectors """
    assert type(smiles) == list
    prod_map = {}
    for ix, prod in enumerate(qm9_grammar.GCFG.productions()):
        prod_map[prod] = ix
    tokenize = molecule_vae.get_qm9_tokenizer(qm9_grammar.GCFG)
    tokens = map(tokenize, smiles)
    parser = nltk.ChartParser(qm9_grammar.GCFG)
    parse_trees = [parser.parse(t).__next__() for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
    return one_hot


def main():
    
    OH = np.zeros((len(L),MAX_LEN,NCHARS))
    for i in range(0, len(L), 100):
        print('Processing: i=[' + str(i) + ':' + str(i+100) + ']')
        onehot = to_one_hot(L[i:i+100])
        OH[i:i+100,:,:] = onehot
    
    h5f = h5py.File('data/grammar_dataset.h5','w')
    h5f.create_dataset('data', data=OH)
    h5f.close()
    
if __name__ == '__main__':
    main()

