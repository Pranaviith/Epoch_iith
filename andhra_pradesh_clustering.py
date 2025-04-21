# Classification Model - Clustering of Pincodes in Andhra Pradesh

# 1. Data Filtering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and filter data
df = pd.read_csv('clustering_data.csv')
df_ap = df[df['StateName'] == 'ANDHRA PRADESH']
df_ap = df_ap[['Pincode', 'Latitude', 'Longitude']].dropna().drop_duplicates()
df_ap['Latitude'] = df_ap['Latitude'].astype(float)
df_ap['Longitude'] = df_ap['Longitude'].astype(float)

# 2. Data Visualization
plt.figure(figsize=(10, 6))
plt.scatter(df_ap['Longitude'], df_ap['Latitude'], s=10, c='blue')
plt.title('Pincode Locations in Andhra Pradesh')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# 3. K-Means Clustering from Scratch
X = df_ap[['Longitude', 'Latitude']].values
k = 4
np.random.seed(0)
centroids = X[np.random.choice(len(X), k, replace=False)]

def assign_clusters(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

for _ in range(10):
    labels = assign_clusters(X, centroids)
    centroids = update_centroids(X, labels, k)

# 4. Cluster Visualization
colors = ['red', 'green', 'blue', 'orange']
plt.figure(figsize=(10, 6))
for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors[i], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=100, label='Centroids')
plt.title('K-Means Clustering of Andhra Pradesh Pincodes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()

# 5. Cluster Summary
cluster_centroids = pd.DataFrame(centroids, columns=['Longitude', 'Latitude'])
cluster_centroids['Cluster'] = [f'Cluster {i+1}' for i in range(k)]
cluster_counts = pd.Series(labels).value_counts().sort_index().reset_index(drop=True)
cluster_centroids['Pincodes Count'] = cluster_counts
print(cluster_centroids)

# 6. Inference and Insights
# - The pincodes are grouped into four clusters representing different regions.
# - Likely aligned with geographic divisions: coast, plains, Rayalaseema, forested areas.
# - Useful for regional logistics, density analysis, and service center distribution.

# 7. Preprocessing Notes
# - Removed duplicates and NaNs
# - Converted coordinates to float
# - Feature standardization skipped due to similar scales

