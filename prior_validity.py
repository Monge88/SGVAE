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

x_can = []
avg = []
count = 0

iterator1 = range(100)
iterator2 = range(100)

with tqdm(iterator1, desc='Prior validity') as molecule_set:
    for i in molecule_set:
        z = np.random.randn(1, 56)
        for j in iterator2:
            x_hat = grammar_model.decode(z)
            
            # check if decodings are valid
            aux = Chem.MolFromSmiles(x_hat[0])
            if aux is not None and aux != '':
            # get canonical representation
                x_can.append(Chem.MolToSmiles(Chem.MolFromSmiles(x_hat[0]), isomericSmiles=True, canonical=True))
                count += 1
        # getting the average for the decoding trials
        avg.append(count/len(iterator2))
        count = 0
        
print(f'Prior validity: {np.mean(avg) * 100:.2f}%')

