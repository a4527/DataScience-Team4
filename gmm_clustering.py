
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

from google.colab import files
uploaded=files.upload()

df = pd.read_csv("power_marketing_dataset_preprocessed.csv")

# construct features manually
df = df.dropna(subset=['Peak Consumption (kWh)', 'Avg Consumption (kWh)', 'Target'])
df['peak_to_avg'] = df['Peak Consumption (kWh)'] / df['Avg Consumption (kWh)']
selected = df[['Incentive Participation_True', 'peak_to_avg', 'Target']]

# standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected)

#  PCA to 3D
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_data)

# fit and plot GMM clustering with different number of clusters
def plot_gmm_clusters(data_3d, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels = gmm.fit_predict(data_3d)
    probs = gmm.predict_proba(data_3d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, s=20, cmap='viridis')
    ax.set_title(f"GMM Clustering (n={n_clusters})")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_box_aspect([20, 20, 20])

    plt.show()

    # Cluster-wise Target mean and size
    result_df = pd.DataFrame(data_3d, columns=['PC1', 'PC2', 'PC3'])
    result_df['Target'] = selected['Target'].values
    result_df['Cluster'] = labels
    print(f"\nCluster Summary for n={n_clusters}:")
    print(result_df.groupby('Cluster')['Target'].agg(['mean', 'count']))

    # samples with ambiguous membership
    max_probs = probs.max(axis=1)
    low_conf_idx = np.where(max_probs < 0.6)[0]
    print(f"
Number of low confidence samples (max prob < 0.6): {len(low_conf_idx)}")
    if len(low_conf_idx) > 0:
        print(result_df.iloc[low_conf_idx].head())

# trying different cluster counts
for n in range(2, 7):
    plot_gmm_clusters(pca_data, n)
