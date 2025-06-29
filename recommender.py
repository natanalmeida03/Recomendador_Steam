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
# Carregamento e Pré-processamento dos dados
# ======================
df = pd.read_csv('./data/steam_enhanced.csv')

numeric_cols_for_fillna = ['price', 'user_engagement', 'price_per_hour', 'playtime_norm', 'price_norm', 'playtime_hours']
for col in numeric_cols_for_fillna:
    df[col] = pd.to_numeric(df[col], errors='coerce') 
    df[col] = df[col].fillna(df[col].median()) 

text_cols = ['genres', 'steamspy_tags']
df[text_cols] = df[text_cols].fillna('')


# ======================
# Criando dataset agregado
# ======================
# Agrupamos por 'clean_game' para ter uma linha única por jogo.
# As funções de agregação (median, mean, unique) consolidam as informações de todas as entradas para o mesmo jogo.
df_agg = df.groupby('clean_game').agg(
    genres=('genres', lambda x: ' '.join(x.dropna().unique())), # Junta gêneros únicos
    steamspy_tags=('steamspy_tags', lambda x: ' '.join(x.dropna().unique())), # Junta tags únicas
    price=('price', 'median'), # Mediana do preço
    user_engagement=('user_engagement', 'mean'), # Média do engajamento
    price_per_hour=('price_per_hour', 'mean'), # Média do preço por hora
    playtime_norm=('playtime_norm', 'mean'), # Média do playtime normalizado
    price_norm=('price_norm', 'mean'), # Média do preço normalizado
    playtime_hours=('playtime_hours', 'median'), # Mediana do playtime para o target 
).reset_index()

# Criar 'text_features' para o DataFrame agregado
df_agg['text_features'] = df_agg['genres'] + ' ' + df_agg['steamspy_tags']

# Definir colunas numéricas para o escalonamento e o Random Forest
num_cols_for_rf = ['price', 'user_engagement', 'price_per_hour', 'playtime_norm', 'price_norm']

# TF-IDF para o DataFrame
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df_agg['text_features'])

# Escalonamento para o DataFrame
scaler = MinMaxScaler()
num_matrix = scaler.fit_transform(df_agg[num_cols_for_rf]) 
num_matrix_sparse = csr_matrix(num_matrix)
final_matrix = hstack([tfidf_matrix, num_matrix_sparse])

# ======================
# Treinar modelo Random Forest com o DataFrame agregado
# ======================
df_agg['target'] = (df_agg['playtime_hours'] > 10).astype(int) 
X = df_agg[num_cols_for_rf]
y = df_agg['target']

if len(X) < 2: 
    warnings.warn("Dados insuficientes para train_test_split. Random Forest pode não ser treinado corretamente.")
    X_train, y_train = X, y 
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, 
        stratify=y if len(y.unique()) > 1 else None
    )

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# ======================
# Função de recomendação com RF (baseado no dataset agregado**)
# ======================
def recomendar_jogos(nome_jogo, top_n=5):
    nome_jogo_lower = nome_jogo.lower()

    if nome_jogo_lower not in df_agg['clean_game'].str.lower().values:
        sugestoes = df_agg[df_agg['clean_game'].str.contains(nome_jogo_lower, na=False, case=False)]['clean_game'].tolist()
        return None, sugestoes

    idx = df_agg[df_agg['clean_game'].str.lower() == nome_jogo_lower].index[0]
    jogo_vec = final_matrix[idx] 

    scores = cosine_similarity(jogo_vec, final_matrix).flatten()

    temp_df = df_agg.copy()
    temp_df['similarity_score'] = scores
    
    candidatos_similares = temp_df[temp_df['clean_game'].str.lower() != nome_jogo_lower].sort_values(by='similarity_score', ascending=False)
    
    candidatos_para_rf = candidatos_similares.head(top_n * 5)

    if candidatos_para_rf.empty:
        return pd.DataFrame(columns=['clean_game', 'genres', 'price', 'rf_prob']), None

    candidatos_para_rf['rf_prob'] = clf.predict_proba(candidatos_para_rf[num_cols_for_rf])[:, 1]
    recomendacoes_finais = candidatos_para_rf.sort_values(by='rf_prob', ascending=False).head(top_n)
    return recomendacoes_finais[['clean_game', 'genres', 'price', 'rf_prob']], None