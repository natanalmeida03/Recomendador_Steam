import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# === 1. Carregar dados ===
df = pd.read_csv('../data/steam_enhanced.csv')

# === 2. Seleção de features e target ===
features = ['price', 'release_year', 'user_engagement', 'price_per_hour', 'price_norm']
X = df[features]
y = df['playtime_hours']

# === 3. Padronização ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 5. Grid Search para Gradient Boosting ===
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 4]
}

grid = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# === 6. Avaliação ===
y_pred = grid.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Melhores Parâmetros:", grid.best_params_)
print("MSE:", mse)
print("MAE:", mae)
print("R²:", r2)
