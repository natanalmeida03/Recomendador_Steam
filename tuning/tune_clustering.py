import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# === 1. Carregar dados ===
df = pd.read_csv('../data/steam_enhanced.csv')

# === 2. Seleção de features numéricas ===
features = ['price', 'user_engagement', 'price_per_hour']
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 3. Testar diferentes valores de K ===
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init='auto')
    model.fit(X_scaled)
    score = silhouette_score(X_scaled, model.labels_)
    silhouette_scores.append(score)

# === 4. Exibir resultados ===
plt.plot(K_range, silhouette_scores, marker='o')
plt.title('Silhouette Score por K')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# === 5. Melhor K ===
best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print("Melhor número de clusters (K):", best_k)
print("Silhouette Score:", max(silhouette_scores))
