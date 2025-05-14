import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados
steam_df = pd.read_csv("../data/steam_enhanced.csv")

# Selecionar features relevantes
features = ["playtime_hours", "price", "positive_ratings", "negative_ratings"]
X = steam_df[features]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduzir dimensão para 2 componentes principais
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Clustering com KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_pca)
labels = kmeans.labels_

# Avaliar clustering
score = silhouette_score(X_pca, labels)
print("=== CLUSTERING ===")
print("Silhouette Score:", round(score, 4))

# Visualização
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="viridis")
plt.title("Clusters de Jogos com PCA + KMeans")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("clustering_plot.png")

