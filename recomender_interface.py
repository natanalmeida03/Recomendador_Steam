import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# ======================
# Pré-processamento
# ======================

# Carregando o dataset
df = pd.read_csv('./data/steam_project.csv')

# Remover duplicatas
df = df.groupby('clean_game').first().reset_index()

# Preencher nulos
text_cols = ['categories', 'genres', 'steamspy_tags']
df[text_cols] = df[text_cols].fillna('')
num_cols = ['required_age', 'achievements', 'positive_ratings', 'negative_ratings', 'average_playtime', 'price']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

# Criar coluna unificada de texto
df['text_features'] = df['categories'] + ' ' + df['genres'] + ' ' + df['steamspy_tags']

# Vetorização TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['text_features'])

# Normalização das features numéricas
scaler = MinMaxScaler()
num_matrix = scaler.fit_transform(df[num_cols])

# Combinar TF-IDF + Numéricas
game_features = hstack([tfidf_matrix, num_matrix])

# ======================
# Função de recomendação
# ======================

def recomendar_jogos(jogo_nome, top_n=5):
    jogo_nome = jogo_nome.strip().lower()
    matches = df[df['clean_game'].str.lower() == jogo_nome]

    if matches.empty:
        print(f"\n❌ Jogo '{jogo_nome}' não encontrado.\n")
        return []

    idx = matches.index[0]
    jogo_vector = game_features.getrow(idx)

    similarity = cosine_similarity(jogo_vector, game_features).flatten()
    similar_indices = similarity.argsort()[::-1]

    # Excluir o próprio jogo e selecionar top_n
    recommendations = [i for i in similar_indices if i != idx][:top_n]
    return df.iloc[recommendations][['clean_game', 'genres', 'price']]

from rich.table import Table
from rich.console import Console

def exibir_recomendacoes(resultados):
    console = Console()
    table = Table(title="Recomendações de Jogos")

    table.add_column("Nome", justify="left", style="cyan", no_wrap=True)
    table.add_column("Gêneros", justify="left", style="magenta")
    table.add_column("Preço (USD)", justify="center", style="green")

    for _, row in resultados.iterrows():
        table.add_row(str(row['clean_game']), str(row['genres']), f"${row['price']:.2f}")

    console.print(table)


# ======================
# Executar
# ======================

if __name__ == "__main__":
    entrada = input("Digite o nome do jogo: ").lower().strip()
    resultados = recomendar_jogos(entrada)
    if not resultados.empty:
        exibir_recomendacoes(resultados)
    else:
        print("Nenhuma recomendação encontrada.")
