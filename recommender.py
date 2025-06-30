import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
from sklearn.cluster import KMeans
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
# Criando dataset agregado (df_agg)
# ======================
df_agg = df.groupby('clean_game').agg(
    genres=('genres', lambda x: ' '.join(x.dropna().unique())),
    steamspy_tags=('steamspy_tags', lambda x: ' '.join(x.dropna().unique())),
    price=('price', 'median'),
    user_engagement=('user_engagement', 'mean'),
    price_per_hour=('price_per_hour', 'mean'),
    playtime_norm=('playtime_norm', 'mean'),
    price_norm=('price_norm', 'mean'),
    playtime_hours=('playtime_hours', 'median'),
).reset_index()

# Criar 'text_features' para o DataFrame agregado
df_agg['text_features'] = df_agg['genres'] + ' ' + df_agg['steamspy_tags']

# ======================
# Treinar K-Means e adicionar cluster ao df_agg
# ======================
kmeans_features = ['price', 'user_engagement', 'price_per_hour', 'playtime_norm', 'price_norm']

kmeans_scaler = MinMaxScaler()
df_agg_kmeans_scaled = kmeans_scaler.fit_transform(df_agg[kmeans_features])

kmeans_model = KMeans(n_clusters=5, random_state=42, n_init=10)
df_agg['game_cluster'] = kmeans_model.fit_predict(df_agg_kmeans_scaled)

# Definir colunas numéricas para o escalonamento da matriz final (TF-IDF + Numéricas)
num_cols_for_final_matrix = ['price', 'user_engagement', 'price_per_hour', 'playtime_norm', 'price_norm']

# TF-IDF para o DataFrame
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df_agg['text_features'])

# Escalonamento para as features numéricas da matriz final (para cosine_similarity)
scaler = MinMaxScaler()
num_matrix = scaler.fit_transform(df_agg[num_cols_for_final_matrix])
num_matrix_sparse = csr_matrix(num_matrix)
final_matrix = hstack([tfidf_matrix, num_matrix_sparse])

# ======================
# Treinar modelo Random Forest para REGRESSÃO
# ======================
# Variável alvo agora é numérica (log-transformada)
df_agg['target_regression'] = np.log1p(df_agg['user_engagement']) # MUDANÇA AQUI

# Features para o Random Forest
num_cols_for_rf = ['price', 'price_per_hour', 'playtime_norm', 'price_norm', 'game_cluster']

X = df_agg[num_cols_for_rf]
y = df_agg['target_regression']
rf_scaler = MinMaxScaler()
X_scaled_for_rf = rf_scaler.fit_transform(X)

if len(X_scaled_for_rf) < 2:
    warnings.warn("Dados insuficientes para train_test_split. Random Forest pode não ser treinado corretamente.")
    X_train, y_train = X_scaled_for_rf, y
else:
    # Sem stratify para regressão
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_for_rf, y, random_state=42)


# Melhores Parâmetros: {'max_depth': 10, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 300}
# RMSE: 0.6539388133522486
# R2 Score: 0.849141001921466
clf = RandomForestRegressor( 
    n_estimators=300, # Depois testar com outros valores (atualmente usando os melhores obtidos no model.py)
    max_depth=10,     
    min_samples_split=2,
    min_samples_leaf=3, 
    random_state=42
)
clf.fit(X_train, y_train)

# ======================
# Função de recomendação com RF
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

    candidatos_para_rf = candidatos_similares.head(top_n * 5).copy()

    if candidatos_para_rf.empty:
        return pd.DataFrame(columns=['clean_game', 'genres', 'price', 'predicted_engagement']), None

    for col in num_cols_for_rf:
        if col not in candidatos_para_rf.columns:
            print(f"Erro: Coluna '{col}' não encontrada em candidatos_para_rf. Verifique a pipeline.")
            return pd.DataFrame(), None
        candidatos_para_rf[col] = pd.to_numeric(candidatos_para_rf[col], errors='coerce').fillna(df_agg[col].median())

    # Escalar as features para previsão usando o mesmo rf_scaler
    candidatos_num_scaled = rf_scaler.transform(candidatos_para_rf[num_cols_for_rf])
    
    predicted_log_engagement = clf.predict(candidatos_num_scaled)
    candidatos_para_rf['predicted_engagement'] = np.expm1(predicted_log_engagement)
    min_eng = df['user_engagement'].min()
    max_eng = df['user_engagement'].max()
    
    # Evitar divisão por zero se max_eng == min_eng
    if max_eng > min_eng:
        candidatos_para_rf['predicted_engagement_norm'] = ((candidatos_para_rf['predicted_engagement'] - min_eng) / (max_eng - min_eng)) * 100
        candidatos_para_rf['predicted_engagement_norm'] = np.clip(candidatos_para_rf['predicted_engagement_norm'], 0, 100)
    else:
        candidatos_para_rf['predicted_engagement_norm'] = 0 # Caso não haja variação

    recomendacoes_finais = candidatos_para_rf.sort_values(by='predicted_engagement', ascending=False).head(top_n)
    return recomendacoes_finais[['clean_game', 'genres', 'price', 'predicted_engagement_norm']], None