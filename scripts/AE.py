import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from extraction import data_dict
# Generate or load your data
# For example purposes, let's create some random data
# data = np.random.rand(1000, 20)  # 1000 samples, 20 features
combined_dataframe = pd.concat(data_dict, ignore_index=True)
data = combined_dataframe.values.astype('float32')
print(data)
# Preprocess data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split data into training and testing sets
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

input_dim = data.shape[1]
encoding_dim = 2  # We want a 2-dimensional encoded representation

# Define the autoencoder model
input_layer = Input(shape=(input_dim,))
encoder = Dense(128, activation='relu')(input_layer)
encoder = Dense(64, activation='relu')(encoder)
encoder_output = Dense(encoding_dim, activation='linear')(encoder)  # Encoded 2D representation

decoder = Dense(64, activation='relu')(encoder_output)
decoder = Dense(128, activation='relu')(decoder)
decoder_output = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder_output)

# Encoder model to get the 2D encoded data
encoder_model = Model(inputs=input_layer, outputs=encoder_output)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=2)

# Encode the test data
encoded_data = encoder_model.predict(X_test)

# Plot the 2D encoded data
plt.figure(figsize=(8, 6))
plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c='blue', marker='o', edgecolor='k')
plt.title('2D Encoded Data')
plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.grid(True)
plt.show()