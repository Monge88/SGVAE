import molecule_vae
import numpy as np
import argparse
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger   
RDLogger.DisableLog('rdApp.*')  


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--epochs', type=int, metavar='N', default=None,    
                        help='Batch size to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=None,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch', type=int, metavar='N', default=None,
                        help='The size of the batch.')
    parser.add_argument('--property', type=str, metavar='N', default=None,
                        help='Property to be used for prediction.')
    return parser.parse_args()

                        
# 1. load grammar VAE
args = get_arguments()
grammar_weights = 'results/vae_grammar_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '_B' + str(args.batch) + '_P_' + str(args.property) + '.hdf5'
grammar_model = molecule_vae.GrammarModel(grammar_weights)

decodings = []
for i in range(100):
    z = np.random.normal(0, 1, (1,56))
    for j in range(100):
        decodings.append(Chem.MolFromSmiles(grammar_model.decode(z)[0]))

decodings[:] = [x for x in decodings if x]  # Returns only valid strings
valid_smiles = [Chem.MolToSmiles(x, isomericSmiles=True, canonical=True) for x in decodings]
novel_molecules = [x for x in valid_smiles if x not in list_df]
        
prior_validity = len(valid_smiles)/len(decodings) * 100
num_novel_molecules = len(novel_molecules)/len(valid_smiles) * 100
num_unique_molecules = len(set(novel_molecules))/len(novel_molecules) * 100

print(f'Prior validity: {prior_validity:.2f}%')
print(f'Percentage of novel molecules % of valid molecules: {num_novel_molecules:.2f}%')
print(f'Percentage of unique molecules: {num_unique_molecules:.2f}%')

