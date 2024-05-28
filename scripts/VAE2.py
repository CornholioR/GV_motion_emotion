import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
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

# Convert the dataframe to a NumPy array
motion_data = combined_dataframe.values.astype('float32')

# Define the encoder architecture
def build_encoder(input_dim, latent_dim, intermediate_dim):
    inputs = Input(shape=(input_dim,))
    hidden = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(hidden)
    encoder = Model(inputs, z_mean)
    return encoder

# Define input dimensions
input_dim = motion_data.shape[1]

# Define latent space dimensions
latent_dim = 2

# Define intermediate dimensions
intermediate_dim = 256

# Build the encoder model
encoder = build_encoder(input_dim, latent_dim, intermediate_dim)

# Compile the encoder model (not necessary for this case)

# Get the encoded representations of the motion data
encoded_data = encoder.predict(motion_data)

# Perform t-SNE on the encoded representations
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