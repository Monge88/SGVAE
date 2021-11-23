import molecule_vae
import argparse
import numpy as np
import make_qm9_dataset_grammar
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

                        
args = get_arguments()
grammar_weights = 'results/vae_grammar_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '_B' + str(args.batch) + '_P_' + str(args.property) + '.hdf5'
grammar_model = molecule_vae.GrammarModel(grammar_weights)
print('Model weights loaded...')

# load molecules for testing 
smiles_test = make_qm9_dataset_grammar.L_test
smiles_test = [Chem.MolToSmiles(Chem.MolFromSmiles(mol), isomericSmiles=True, canonical=True) for mol in smiles_test]

aux = 0
avg = []
with tqdm(smiles_test, desc='Reconstruction accuracy') as molecule_set:
    for molecule in molecule_set:
        for i in range(10):
            z = grammar_model.encode([molecule])
            for j in range(10):
                x_hat = grammar_model.decode(z)
                if x_hat[0] == molecule:
                    aux += 1
        mol_avg = aux/100
        avg.append(mol_avg)
        aux = 0

    
print(f'The percentage of correct reconstruction is {np.mean(avg) * 100:.2f} %')


