import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K 

from extraction import data_dict

# Load your motion dataframes (assuming you have already loaded them)
# Replace this with loading your dataframes
# For example:
# dataframe1 = pd.read_csv('motion_data1.csv')
# dataframe2 = pd.read_csv('motion_data2.csv')
# ...

# Combine the dataframes into a single dataframe
combined_dataframe = pd.concat(data_dict, ignore_index=True)

# Preprocess the data if necessary (e.g., normalize)

# Define the VAE architecture
def build_vae(input_dim, latent_dim, intermediate_dim):
    # Encoder
    inputs = Input(shape=(input_dim,))
    hidden = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)
    
    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling)([z_mean, z_log_var])
    
    # Define the VAE model
    vae = Model(inputs, z_mean)
    
    # Define the VAE loss
    reconstruction_loss = mse(inputs, z_mean)
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    
    return vae

# Define input dimensions
input_dim = combined_dataframe.shape[1]

# Define latent space dimensions
latent_dim = 2

# Define intermediate dimensions
intermediate_dim = 256

# Build the VAE model
vae = build_vae(input_dim, latent_dim, intermediate_dim)

# Compile the VAE model
vae.compile(optimizer='adam')

# Train the VAE model
history = vae.fit(combined_dataframe, epochs=50, batch_size=32, validation_split=0.2)

# Get the latent representations of the motion data
encoded_data = vae.predict(combined_dataframe)

# Perform t-SNE on the latent representations
tsne = TSNE(n_components=2)
tsne_data = tsne.fit_transform(encoded_data)

# Plot the latent space using t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c='blue', alpha=0.5)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('Latent Space Visualization using t-SNE')
plt.grid(True)
plt.show()
