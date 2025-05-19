import streamlit as st
from recommender import load_and_preprocess_data, recomendar_jogos

# Configurar a página do Streamlit
st.set_page_config(page_title="Game Recommender", page_icon="🎮", layout="wide")

# Título e descrição
st.title("🎮 Game Recommendation System")
st.markdown("Select a game from the dropdown below to get personalized game recommendations based on genres, categories, and other features.")

# Carregar dados
df, game_features = load_and_preprocess_data()

# Interface do Streamlit
with st.form("recommendation_form"):
    st.subheader("Select a Game")
    game_name = st.selectbox("Choose a game:", options=[''] + sorted(df['clean_game'].tolist()), index=0)
    top_n = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)
    submit_button = st.form_submit_button("Get Recommendations")

# Exibir recomendações
if submit_button and game_name:
    st.subheader(f"Recommendations for {game_name}")
    recommendations = recomendar_jogos(game_name, df, game_features, top_n)
    
    if recommendations:
        for i, game in enumerate(recommendations, 1):
            st.write(f"{i}. {game}")
    else:
        st.error(f"Game '{game_name}' not found. Please select a valid game from the dropdown.")
else:
    st.info("Please select a game and click 'Get Recommendations' to see results.")