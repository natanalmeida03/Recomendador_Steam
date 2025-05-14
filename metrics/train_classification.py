import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Carrega o dataset tratado
df = pd.read_csv("../data/steam_enhanced.csv")

# Filtrar apenas os usuários que jogaram mais de 10 horas para criar uma classe
threshold = 10
df["target"] = (df["playtime_hours"] > threshold).astype(int)

# Selecionar features
features = ["price", "positive_ratings", "negative_ratings", "price_per_hour"]
X = df[features]
y = df["target"]

# Dividir o conjunto de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Avaliar modelo
y_pred = clf.predict(X_test)
print("=== CLASSIFICAÇÃO ===")
print("Acurácia:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

