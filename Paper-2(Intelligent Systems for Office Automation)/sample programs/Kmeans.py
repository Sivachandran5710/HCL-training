# %% 
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# %% 
# 2. Load dataset
# Make sure Mall_Customers.csv is in the same folder as this script
df = pd.read_csv("Mall_Customers.csv")
print("Dataset Loaded Successfully ")
print(df.head())

# %% 
# 3. Exploratory Data Analysis (EDA)
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

# Plot histograms for numeric columns
df.hist(bins=20, figsize=(12,6), color="skyblue", edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=14)
plt.show()

# Correlation heatmap (numeric only)
plt.figure(figsize=(6,4))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# %% 
# 4. Data Preprocessing
print("Column Names in Dataset:", df.columns)

# Drop non-numeric columns
if 'Gender' in df.columns:
    X = df.drop(['CustomerID','Gender'], axis=1)
elif 'Genre' in df.columns:
    X = df.drop(['CustomerID','Genre'], axis=1)
else:
    X = df.drop(['CustomerID'], axis=1)  # fallback

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled Data Shape:", X_scaled.shape)


# %% 
# 5. PCA (reduce to 2D for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

plt.scatter(X_pca[:,0], X_pca[:,1], c="gray", edgecolor="k")
plt.title("PCA projection of Customers (unclustered)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# %% 
# 6. Elbow Method (find best k)
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal k")
plt.show()

# %% 
# 7. Apply KMeans (choose k=5 for this dataset)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

print("\nSample Cluster Assignments:")
print(df[['CustomerID','Age','Annual Income (k$)','Spending Score (1-100)','Cluster']].head())

# %% 
# 8. Evaluate Clustering
silhouette_avg = silhouette_score(X_scaled, clusters)
print("\nSilhouette Score:", silhouette_avg)

# %% 
# 9. Visualize clusters in PCA space
plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='plasma', s=60, edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            s=200, c='red', marker='X', label='Centroids')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans Clustering of Customers")
plt.legend()
plt.show()

# %% 
# 10. Analyze clusters
cluster_summary = df.groupby("Cluster")[["Age","Annual Income (k$)","Spending Score (1-100)"]].mean()
print("\n--- Cluster Summary ---")
print(cluster_summary)
