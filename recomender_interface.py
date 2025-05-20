import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
from rich.console import Console
from rich.table import Table
import warnings
warnings.filterwarnings("ignore")

# ======================
# Carregamento e Preparo
# ======================

# Carregar dataset
df = pd.read_csv('./data/steam_enhanced.csv')

# Remover duplicatas
df = df.drop_duplicates(subset='clean_game', keep='first').reset_index(drop=True)

# Preencher nulos
num_cols = ['price', 'user_engagement', 'price_per_hour', 'playtime_norm', 'price_norm']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(df[num_cols].median())
text_cols = ['genres', 'steamspy_tags']
df[text_cols] = df[text_cols].fillna('')

# Combinar colunas textuais
df['text_features'] = df['genres'] + ' ' + df['steamspy_tags']

# TF-IDF com n-gramas
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df['text_features'])

# Normalizar colunas numéricas
scaler = MinMaxScaler()
num_matrix = scaler.fit_transform(df[num_cols])
num_matrix_sparse = csr_matrix(num_matrix)

# Combinar todas as features
game_features = hstack([tfidf_matrix, num_matrix_sparse])

# ======================
# Função de Recomendação
# ======================

def recomendar_jogos(nome_jogo, top_n=5):
    nome_jogo = nome_jogo.lower()
    if nome_jogo not in df['clean_game'].values:
        print(f"\n❌ Jogo '{nome_jogo}' não encontrado.")
        similares = df[df['clean_game'].str.contains(nome_jogo, na=False)]['clean_game'].tolist()
        if similares:
            print("🔍 Você quis dizer:")
            for s in similares[:5]:
                print(f" - {s}")
        return pd.DataFrame()

    idx = df[df['clean_game'] == nome_jogo].index[0]
    jogo_vector = game_features[idx]
    similarity = cosine_similarity(jogo_vector, game_features).flatten()

    similar_indices = similarity.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n]

    return df.iloc[similar_indices][['clean_game', 'genres', 'price']]

# ======================
# Interface com Rich
# ======================

def exibir_recomendacoes(recomendados):
    console = Console()
    table = Table(title="🎮 Recomendação de Jogos Steam")

    table.add_column("Nome", style="cyan", no_wrap=True)
    table.add_column("Gêneros", style="magenta")
    table.add_column("Preço", justify="center", style="green")

    for _, row in recomendados.iterrows():
        table.add_row(str(row['clean_game']), row['genres'], f"${row['price']:.2f}")

    console.print(table)

# ======================
# Execução
# ======================

if __name__ == "__main__":
    nome = input("Digite o nome de um jogo: ").strip().lower()
    recomendados = recomendar_jogos(nome)
    
    if not recomendados.empty:
        exibir_recomendacoes(recomendados)
    else:
        print("\nNenhuma recomendação pôde ser exibida.")
