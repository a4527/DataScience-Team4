import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from google.colab import files

uploaded = files.upload()

# load dataset
df = pd.read_csv("power_marketing_dataset_preprocessed.csv")

# select features for PCA
pca_features = df[['Marketing Interaction_True', 'Incentive Participation_True']].dropna()
pca = PCA(n_components=1)
pca_components = pca.fit_transform(pca_features)

# reflect PCA result into original dataframe
df = df.loc[pca_features.index].copy()
df['pca'] = pca_components[:, 0]

# select features for clustering
clustering_vars = ['pca', 'Peak Consumption (kWh)', 'Target']
df_clustering = df[clustering_vars].dropna().copy()

# loop through n = 2 to 6
for n in range(2, 7):
    X = df_clustering[['pca', 'Peak Consumption (kWh)', 'Target']].copy()
    gmm = GaussianMixture(n_components=n, random_state=42)
    df_clustering[f'gmm_cluster_{n}'] = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)

    # 3D Visualization
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        df_clustering['pca'],
        df_clustering['Peak Consumption (kWh)'],
        df_clustering['Target'],
        c=df_clustering[f'gmm_cluster_{n}'],
        cmap='plasma', s=30
    )
    ax.set_title(f"GMM Clustering (n={n})")
    ax.set_xlabel('PCA (Participation Tendency)')
    ax.set_ylabel('Peak Consumption (kWh)')
    ax.set_zlabel('Target (Energy Reduction)')
    plt.tight_layout()
    plt.show()

    # Cluster-wise Target mean and sample count
    print(f"\nCluster Summary for n={n}:")
    summary = df_clustering.groupby(f'gmm_cluster_{n}')['Target'].agg(['mean', 'count'])
    print(summary)

    # Low confidence samples (max probability < 0.6)
    max_probs = probs.max(axis=1)
    low_conf_idx = np.where(max_probs < 0.6)[0]
    print(f"Number of low confidence samples (max prob < 0.6): {len(low_conf_idx)}")
    if len(low_conf_idx) > 0:
        print(df_clustering.iloc[low_conf_idx][['pca', 'Peak Consumption (kWh)', 'Target']].head())
