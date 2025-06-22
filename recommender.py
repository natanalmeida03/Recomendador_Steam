import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# ======================
# Carregamento dos dados
# ======================
df = pd.read_csv('./data/steam_enhanced.csv')
df = df.drop_duplicates(subset='clean_game', keep='first').reset_index(drop=True)

num_cols = ['price', 'user_engagement', 'price_per_hour', 'playtime_norm', 'price_norm']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

text_cols = ['genres', 'steamspy_tags']
df[text_cols] = df[text_cols].fillna('')
df['text_features'] = df['genres'] + ' ' + df['steamspy_tags']

tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['text_features'])

scaler = MinMaxScaler()
num_matrix = scaler.fit_transform(df[num_cols])
num_matrix_sparse = csr_matrix(num_matrix)
final_matrix = hstack([tfidf_matrix, num_matrix_sparse])

# ======================
# Treinar modelo Random Forest
# ======================
df['target'] = (df['playtime_hours'] > 10).astype(int)
X = df[num_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# ======================
# Função de recomendação com RF
# ======================
def recomendar_jogos(nome_jogo, top_n=5):
    nome_jogo = nome_jogo.lower()
    if nome_jogo not in df['clean_game'].values:
        sugestões = df[df['clean_game'].str.contains(nome_jogo, na=False)]['clean_game'].tolist()
        return None, sugestões

    idx = df[df['clean_game'] == nome_jogo].index[0]
    jogo_vec = final_matrix[idx]
    scores = cosine_similarity(jogo_vec, final_matrix).flatten()

    similar_indices = np.argsort(scores)[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n * 2]

    candidatos = df.iloc[similar_indices].copy()
    candidatos['rf_prob'] = clf.predict_proba(candidatos[num_cols])[:, 1]
    candidatos = candidatos.sort_values(by='rf_prob', ascending=False).head(top_n)

    return candidatos[['clean_game', 'genres', 'price', 'rf_prob']], None