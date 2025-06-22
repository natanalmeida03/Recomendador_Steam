import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# Carregar o dataset
df = pd.read_csv('./data/steam_enhanced.csv')

# Tratar duplicatas
df = df.drop_duplicates(subset='clean_game', keep='first').reset_index(drop=True)

# Preenchimento de nulos
num_cols = ['price', 'user_engagement', 'price_per_hour', 'playtime_norm', 'price_norm']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

text_cols = ['genres', 'steamspy_tags']
df[text_cols] = df[text_cols].fillna('')

# Criar feature textual unificada
df['text_features'] = df['genres'] + ' ' + df['steamspy_tags']

# TF-IDF com bigramas
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['text_features'])

# Normalizar features numéricas
scaler = MinMaxScaler()
num_matrix = scaler.fit_transform(df[num_cols])

# Concatenar matrizes
from scipy.sparse import csr_matrix
num_matrix_sparse = csr_matrix(num_matrix)
final_matrix = hstack([tfidf_matrix, num_matrix_sparse])

# Função de recomendação
def recomendar_jogos(nome_jogo, top_n=5):
    nome_jogo = nome_jogo.lower()
    if nome_jogo not in df['clean_game'].values:
        print(f"[ERRO] Jogo '{nome_jogo}' não encontrado.")
        sugestões = df[df['clean_game'].str.contains(nome_jogo, na=False)]['clean_game'].tolist()
        if sugestões:
            print("Você quis dizer:")
            for sug in sugestões[:5]:
                print(f" - {sug}")
        return []
    
    idx = df[df['clean_game'] == nome_jogo].index[0]
    jogo_vec = final_matrix[idx]
    scores = cosine_similarity(jogo_vec, final_matrix).flatten()
    
    similar_indices = np.argsort(scores)[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n]
    
    print(f"\n🎮 Recomendados para '{df.loc[idx, 'clean_game']}':\n")
    return df.iloc[similar_indices][['clean_game', 'genres', 'price']]

# Exemplo
recomendados = recomendar_jogos("portal 2", top_n=5)
print(recomendados)
