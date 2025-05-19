import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

def load_and_preprocess_data():
    # Carregar o dataset
    df = pd.read_csv('./data/steam_project.csv')

    # Verificar duplicatas em 'clean_game'
    duplicates = df['clean_game'].duplicated().sum()
    if duplicates > 0:
        df = df.groupby('clean_game').first().reset_index()

    # Tratar valores nulos
    num_cols = ['required_age', 'achievements', 'positive_ratings', 'negative_ratings', 'average_playtime', 'price']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median()).astype('float32')
    text_cols = ['categories', 'genres', 'steamspy_tags']
    df[text_cols] = df[text_cols].fillna('')

    # Criar coluna com textos combinados
    df['text_features'] = df['categories'] + ' ' + df['genres'] + ' ' + df['steamspy_tags']

    # TF-IDF nas features de texto
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['text_features'])

    # Normalizar features numéricas
    scaler = MinMaxScaler()
    num_features = scaler.fit_transform(df[num_cols]).astype('float32')

    # Concatenar TF-IDF e numéricas
    game_features = hstack([tfidf_matrix, num_features])

    return df, game_features

def recomendar_jogos(jogo_nome, df, game_features, top_n=5):
    try:
        idx = df[df['clean_game'] == jogo_nome].index[0]
        jogo_vector = game_features.getrow(idx)
        similarity_scores = cosine_similarity(jogo_vector, game_features).flatten()
        similar_indices = np.argsort(similarity_scores)[::-1]
        similar_indices = [i for i in similar_indices if i != idx][:top_n]
        return df.iloc[similar_indices]['clean_game'].tolist()
    except IndexError:
        return []