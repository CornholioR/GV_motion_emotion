import seaborn as sns
import numpy as np
from extraction import dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from extraction import emotionData
import matplotlib.pyplot as plt

data_train = dataset #(3766, 60, 6)
labels = emotionData #list of strings with the emotions
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_samples = data_train.shape[0]
def autocorrelation_matrices_sample(sample):
    num_timesteps, _ = sample.shape
    autocorrelation_matrices_sample = []
    for k in range(num_timesteps):
        R = sample[k,].reshape(-1,1) @ sample[k-1,].reshape(1,-1)
        autocorrelation_matrices_sample.append(R)
    return np.array(autocorrelation_matrices_sample)
feature_vectors = []
for i in range(num_samples):
    autocorrelation_matrices_sample_var = autocorrelation_matrices_sample(data_train[i,:,:])
    autocorrelation_matrices_mean = np.mean(autocorrelation_matrices_sample_var, axis=0)
    feature_vectors.append(autocorrelation_matrices_mean.flatten())
feature_vectors = np.array(feature_vectors)
print(feature_vectors.shape)
pca = PCA()
pca.fit(feature_vectors)
transformed_data = pca.transform(feature_vectors)
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels_encoded[:], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on feature vectors obtained after auto-correlation')
plt.colorbar(label='Feature Index')
plt.grid(True)
plt.show()