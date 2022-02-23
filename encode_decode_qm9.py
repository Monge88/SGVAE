import molecule_vae
import os.path
import h5py
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import argparse
import pickle
import make_qm9_dataset_grammar
import prop_prediction_model
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Draw
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
        

# load grammar VAE
np.random.seed(1)
args = get_arguments()
grammar_weights = 'results/vae_grammar_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '_B' + str(args.batch) + '_P_' + str(args.property) + '.hdf5'
grammar_model = molecule_vae.GrammarModel(grammar_weights)

# loading the training and the test data
df = make_qm9_dataset_grammar.df
smiles_test = make_qm9_dataset_grammar.L_test
smiles_train = make_qm9_dataset_grammar.L
ids_train = make_qm9_dataset_grammar.ids_train
ids_test = make_qm9_dataset_grammar.ids_test

# encoding the training molecules
print('Encoding training molecules')
z_train = grammar_model.encode(smiles_train)
# encoding the testing molecules
print('Encoding testting molecules')
z = grammar_model.encode(smiles_test)

# visualizing the latent space

if args.latent_dim == 2:
    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], c=df[args.property][ids_train[:30000]], cmap='viridis', s=5)
    plt.colorbar()
    plt.show()

else:
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(z)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df[args.property][ids_train[:30000]], cmap='viridis', s=5)
    plt.colorbar()
    plt.show()


scaler = StandardScaler()
# scaling and dividing the property values by the number of atoms in each molecule
n_atoms = np.array(df['number_of_atoms'][ids_train])
prop_real = np.array(df[args.property][ids_train])
prop_real_div = np.array(prop_real/n_atoms)
prop_norm = np.array(scaler.fit_transform(prop_real_div.reshape(-1, 1)))
# training and saving the property prediction model
prop_prediction_model.neural_model(z_train, prop_norm, args.property)

# testing the property prediction model
prediction_weights = 'prop_pred/prediction_weights_' + args.property + '_.h5'
pp_model = load_model(prediction_weights)

n_atoms = np.array(df['number_of_atoms'][ids_test])
prop_real = np.array(df[args.property][ids_test])
prop_real_div = np.array(prop_real/n_atoms)
prop_norm = np.array(scaler.fit_transform(prop_real_div.reshape(-1, 1)))

pp_error = pp_model.evaluate(z, prop_norm)
print(f'Prediction error: {pp_error} eV')
