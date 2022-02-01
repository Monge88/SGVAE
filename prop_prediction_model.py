import os
import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split, KFold

def neural_model(x_lat, y_prop, prop_name):
    
    np.random.seed(1)

    x, x_test, y, y_test = train_test_split(x_lat, y_prop, test_size=0.1)
 
    score = []
    pred = []
    y_pred = []
    kfold = KFold(n_splits=10, shuffle=True)
    
    input_layer = Input(shape=(x.shape[1],))

    h = Dense(512, activation='relu', name='prop_1')(input_layer)
    h = Dropout(0.2)(h)
    h = Dense(512, activation='relu', name='prop_2')(h)
    h = Dropout(0.3)(h)
    h = BatchNormalization()(h)
    h_out = Dense(1, activation='linear', name='prop_out')(h)

    model = Model(input_layer, h_out)
    model.compile(loss='mean_absolute_error',
                  optimizer='adam')
                  
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 5,
                                  min_lr = 0.0001)
    
    for train, validation in kfold.split(x, y):
    
        model.fit(x[train], y[train], 
                  epochs=100,
                  batch_size=256,
                  verbose=0,
                  callbacks=[reduce_lr],
                  validation_data=(x[validation], y[validation]))
              
    folder = os.path.join('prop_pred_weights', prop_name)    
    if not os.path.exists(folder):
    	os.makedirs(folder)
        
        
    model.save(os.path.join(folder, 'prediction_weights.h5'))
