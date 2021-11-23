# Supervised Grammar Variational Autoencoder
This repository contains the code used in the paper: \
This code was inspired in the code for the paper [Grammar Variational Autoencoder](https://arxiv.org/abs/1703.01925), which can be found at https://github.com/mkusner/grammarVAE.

### Creating the dataset
To create the molecular dataset, use:
* ```python make_qm9_dataset_grammar.py```

### Training the model
When training the model you can specify the number of epochs, batch size, latent space dimension and the property to be used. Call:
* ```python model.py --epochs=100 --batch=256 --latent_dim=56 --property=energy_of_LUMO``` \

The name of the property should match the columns names in the ```QM9_STAR.pkl``` data file. To check the column names in the data file, use:
```
import pickle
with open("data/QM9_STAR.pkl", "rb") as data:
    df = pickle.load(data) 
    print(df.columns)
```

### Testing the model
To plot the latent space, train and test the property prediction model, call:
* ```python encode_decode_qm9.py --epochs=100 --batch=256 --latent_dim=56 --property=energy_of_LUMO``` \

Notice that you have to specify the same arguments used to train the model, as they are used in the weights' file name. \
Finally, to test the models' prior validity and reconstruction accuracy, call:
* ```python prior_validity.py --epochs=100 --batch=256 --latent_dim=56 --property=energy_of_LUMO``` 
* ```python reconstruction.py --epochs=100 --batch=256 --latent_dim=56 --property=energy_of_LUMO``` 
