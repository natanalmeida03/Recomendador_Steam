import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# === 1. Carregar dados ===
df = pd.read_csv('./data/steam_enhanced.csv')

numeric_cols_for_fillna = ['price', 'user_engagement', 'price_per_hour', 'playtime_norm', 'price_norm', 'playtime_hours']
for col in numeric_cols_for_fillna:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Replicar a agregação para garantir consistência
df_agg_for_tuning = df.groupby('clean_game').agg(
    price=('price', 'median'),
    user_engagement=('user_engagement', 'mean'),
    price_per_hour=('price_per_hour', 'mean'),
    playtime_norm=('playtime_norm', 'mean'),
    price_norm=('price_norm', 'mean'),
    playtime_hours=('playtime_hours', 'median'),
).reset_index()

# === 2. Treinar K-Means e adicionar cluster como feature ===
kmeans_features = ['price', 'user_engagement', 'price_per_hour', 'playtime_norm', 'price_norm']
kmeans_scaler = MinMaxScaler()
df_agg_kmeans_scaled = kmeans_scaler.fit_transform(df_agg_for_tuning[kmeans_features])
kmeans_model_for_tuning = KMeans(n_clusters=5, random_state=42, n_init=10)
df_agg_for_tuning['game_cluster'] = kmeans_model_for_tuning.fit_predict(df_agg_kmeans_scaled)

# === 3. Criar variável alvo numérica (Para regressão) ===
df_agg_for_tuning['target_regression'] = np.log1p(df_agg_for_tuning['user_engagement']) #Target numérica e log-transformada

# === 4. Seleção de features para o Random Forest ===
features = ['price', 'price_per_hour', 'playtime_norm', 'price_norm', 'game_cluster']
X = df_agg_for_tuning[features]
y = df_agg_for_tuning['target_regression'] 

# === 5. Escalonamento para o Random Forest ===
rf_scaler = MinMaxScaler()
X_scaled = rf_scaler.fit_transform(X)

# === 6. Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) 

# === 7. GridSearchCV para ajuste fino ===
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

# RandomForestRegressor e scoring para regressão (neg_mean_squared_error ou r2)
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

# === 8. Avaliação ===
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Melhores Parâmetros:", grid.best_params_)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))
