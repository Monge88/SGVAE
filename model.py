import make_qm9_dataset_grammar
import argparse
import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import qm9_grammar as G
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, BatchNormalization, Activation, Flatten, RepeatVector, TimeDistributed, GRU, Convolution1D
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# helper variables in Keras format for parsing the grammar
masks_K      = K.variable(G.masks)
ind_of_ind_K = K.variable(G.ind_of_ind)

MAX_LEN = 100
DIM = G.D


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--load_model', type=str, metavar='N', default="")
    parser.add_argument('--epochs', type=int, metavar='N', default=100,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=None,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--batch', type=int, metavar='N', default=256,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--property', type=str, metavar='N', default=None,
                        help='Property to be used to shape the latent representation.')
    return parser.parse_args()


class MoleculeVAE():

    autoencoder = None
    merged = None
    
    def create(self,
               charset,
               max_length = MAX_LEN,
               latent_rep_size = 2,
               weights_file = None):
        charset_length = len(charset)
        
        x = Input(shape=(max_length, charset_length), name='autoencoder_input')
        _, z, mlp = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = Model(x, [z, mlp])

        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        x1 = Input(shape=(max_length, charset_length), name='autoencoder_input1')
        vae_loss, z1, mlp_model = self._buildEncoder(x1, latent_rep_size, max_length)
        self.autoencoder = Model(
            inputs = x1,
            outputs=[
            self._buildDecoder(
                z1,
                latent_rep_size,
                max_length,
                charset_length
            ), mlp_model]
        )

        # for obtaining mean and log variance of encoding distribution
        x2 = Input(shape=(max_length, charset_length))
        (z_m, z_l_v) = self._encoderMeanVar(x2, latent_rep_size, max_length)
        self.encoderMV = Model(x2, [z_m, z_l_v])

        if weights_file:
            self.autoencoder.load_weights(weights_file, by_name=True)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)
            self.encoderMV.load_weights(weights_file, by_name = True)

        self.autoencoder.compile(optimizer = 'adam',
                                 loss = {'decoded_mean': vae_loss,
                                        'mlp_out': tf.keras.losses.mse}
                                        )
                                         
        
    def _encoderMeanVar(self, x, latent_rep_size, max_length, epsilon_std = 0.01):
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation = 'relu', name='dense_1')(h)

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        return (z_mean, z_log_var) 
        

    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 1):
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation = 'relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        
        def conditional(x_true, x_pred):
            most_likely = K.argmax(x_true)
            most_likely = tf.reshape(most_likely,[-1]) 
            ix2 = tf.expand_dims(tf.gather(ind_of_ind_K, most_likely),1)
            ix2 = tf.cast(ix2, tf.int32)  
            M2 = tf.gather_nd(masks_K, ix2) # get slices of masks_K with indices
            M3 = tf.reshape(M2, [-1,MAX_LEN,DIM]) # reshape them
            P2 = tf.multiply(K.exp(x_pred),M3) # apply them to the exp-predictions
            P2 = tf.divide(P2,K.sum(P2,axis=-1,keepdims=True)) # normalize predictions
            return P2          
            

        def vae_loss(x, x_decoded_mean):
            x_decoded_mean = conditional(x, x_decoded_mean) # we add this new function to the loss
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
   
            return xent_loss + kl_loss
        
        encoded_out = Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])
        
        mlp = Dense(70, activation='relu', name='mlp_1')(encoded_out)
        mlp = Dense(70, activation='relu', name='mlp_2')(mlp)
        mlp = Dense(70, activation='relu', name='mlp_3')(mlp)
        mlp = BatchNormalization()(mlp)

        mlp = Dense(1, activation='linear', name='mlp_out')(mlp)
        
        return (vae_loss, encoded_out, mlp)

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(500, return_sequences = True, reset_after=False, name='gru_1')(h)
        h = GRU(500, return_sequences = True, reset_after=False, name='gru_2')(h)
        h = GRU(500, return_sequences = True, reset_after=False, name='gru_3')(h)
        return TimeDistributed(Dense(charset_length), name='decoded_mean')(h)
        
    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, latent_rep_size = 56, max_length=MAX_LEN):
        self.create(charset, max_length = max_length, weights_file = weights_file, latent_rep_size = latent_rep_size)


if __name__ == '__main__':
    
    rules = G.gram.split('\n')
    
    # load dataset
    h5f = h5py.File('data/grammar_dataset.h5', 'r')
    data = h5f['data'][:]
    h5f.close()

    # get any arguments and define save file, then create the VAE model
    np.random.seed(1)
    args = get_arguments()

    if not os.path.exists('results'):
    	os.mkdir('results')
    	
    model_save = 'results/vae_grammar_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '_B' + str(args.batch) + '_P_' + str(args.property) + '.hdf5'
        
    print(model_save)
    model = MoleculeVAE()
    print(args.load_model)

    # if this results file exists already load it
    if os.path.isfile(args.load_model):
        print('loading!')
        model.load(rules, args.load_model, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
    else:
        print('making new model')
        model.create(rules, max_length=MAX_LEN, latent_rep_size = args.latent_dim)

    
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)
                                  
    model_checkpoint = ModelCheckpoint(model_save, 
                                       verbose=1, 
                                       save_best_only=True)
                                  
    # loading the dataset with the property values and ids for training
    df = make_qm9_dataset_grammar.df
    ids_train = make_qm9_dataset_grammar.ids_train
    
    # normalizing the data and dividing it by the number of atoms in a molecule
    scaler = StandardScaler()
    prop = np.array(df[args.property][ids_train]/df['number_of_atoms'][ids_train])
    prop_norm = np.array(scaler.fit_transform(prop.reshape(-1, 1)))


    model.autoencoder.fit(
        {'autoencoder_input1': data},
        {'decoded_mean': data, 'mlp_out':prop_norm},
        shuffle = True,
        epochs = args.epochs,
        batch_size = args.batch,
        callbacks = [reduce_lr, model_checkpoint],
        validation_split = 0.01
        )
    
