import streamlit as st
st.set_page_config(page_title="🎮 Game Recommender", layout="wide")  # PRIMEIRA LINHA Streamlit

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# ======================
# Carregamento e Preparo
# ======================

@st.cache_resource
def load_data():
    df = pd.read_csv('./data/steam_enhanced.csv')

    # Remover duplicatas
    df = df.drop_duplicates(subset='clean_game', keep='first').reset_index(drop=True)

    # Preencher nulos
    num_cols = ['price', 'user_engagement', 'price_per_hour', 'playtime_norm', 'price_norm']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    text_cols = ['genres', 'steamspy_tags']
    df[text_cols] = df[text_cols].fillna('')
    df['text_features'] = df['genres'] + ' ' + df['steamspy_tags']

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['text_features'])

    # Normalizar features numéricas
    scaler = MinMaxScaler()
    num_matrix = scaler.fit_transform(df[num_cols])
    num_matrix_sparse = csr_matrix(num_matrix)

    # Matriz final
    final_matrix = hstack([tfidf_matrix, num_matrix_sparse])

    return df, final_matrix

df, game_features = load_data()

# ======================
# Função de Recomendação
# ======================

def recomendar_jogos(nome_jogo, top_n=5):
    nome_jogo = nome_jogo.lower()
    if nome_jogo not in df['clean_game'].values:
        sugestões = df[df['clean_game'].str.contains(nome_jogo, na=False)]['clean_game'].tolist()
        return None, sugestões

    idx = df[df['clean_game'] == nome_jogo].index[0]
    jogo_vec = game_features[idx]
    scores = cosine_similarity(jogo_vec, game_features).flatten()

    similar_indices = np.argsort(scores)[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n]

    return df.iloc[similar_indices][['clean_game', 'genres', 'price']], None

# ======================
# Interface Streamlit
# ======================

st.title("🎮 Recomendador de Jogos Steam")
st.markdown("Escolha um jogo e receba recomendações personalizadas com base em gêneros e características similares.")

with st.form("form_recomender"):
    game_name = st.selectbox("Selecione um jogo:", options=[''] + sorted(df['clean_game'].tolist()), index=0)
    top_n = st.slider("Quantidade de recomendações", min_value=1, max_value=10, value=5)
    submit = st.form_submit_button("Recomendar")

if submit and game_name:
    recomendacoes, sugestoes = recomendar_jogos(game_name, top_n=top_n)

    if recomendacoes is not None:
        st.subheader(f"🔎 Recomendações para: **{game_name}**")
        for i, row in recomendacoes.iterrows():
            st.markdown(f"**{row['clean_game'].title()}**  \n🎯 *Gêneros:* {row['genres']}  \n💲 *Preço:* ${row['price']:.2f}")
            st.markdown("---")
    elif sugestoes:
        st.warning(f"Jogo '{game_name}' não encontrado exatamente. Você quis dizer:")
        for sug in sugestões[:5]:
            st.markdown(f"- {sug}")
    else:
        st.error("Nenhum jogo correspondente foi encontrado.")
