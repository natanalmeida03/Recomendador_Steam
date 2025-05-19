import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# === 1. Carregar dados ===
df = pd.read_csv('../data/steam_enhanced.csv')

# === 2. Criar variável alvo binária ===
df['high_engagement'] = (df['user_engagement'] > df['user_engagement'].median()).astype(int)

# === 3. Seleção de features e target ===
features = ['price', 'release_year', 'price_per_hour', 'playtime_norm', 'price_norm']
X = df[features]
y = df['high_engagement']

# === 4. Padronização ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 5. Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 6. GridSearchCV para ajuste fino ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

# === 7. Avaliação ===
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Melhores Parâmetros:", grid.best_params_)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
