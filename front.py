import streamlit as st
import pandas as pd
from recommender import recomendar_jogos

# ===========================
# Configuração da Página
# ===========================
st.set_page_config(
    page_title="🎮 Steam Game Recommender",
    page_icon="🎮",
    layout="wide"
)

# ===========================
# Cabeçalho
# ===========================
st.title("🎮 Steam Game Recommender")
st.markdown(
    """
    Obtenha recomendações personalizadas de jogos da Steam com base em **conteúdo textual (TF-IDF + Cosine Similarity)** e um modelo **Random Forest** que estima sua probabilidade de gostar de cada jogo.
    """
)

# ===========================
# Formulário
# ===========================
with st.form("recommender_form"):
    search_text = st.text_input("🔍 Digite parte do nome do jogo:", "")
    top_n = st.slider("📋 Número de recomendações:", 1, 10, 5)
    submit = st.form_submit_button("Recomendar")

# ===========================
# Recomendação
# ===========================
if submit and search_text.strip():
    st.subheader("🎯 Resultados")

    recomendacoes, sugestoes = recomendar_jogos(search_text.lower().strip(), top_n)

    if recomendacoes is not None and not recomendacoes.empty:
        num_columns = 2
        rows = [recomendacoes.iloc[i:i+num_columns] for i in range(0, len(recomendacoes), num_columns)]

        for row in rows:
            cols = st.columns(num_columns)
            for i, (_, jogo) in enumerate(row.iterrows()):
                with cols[i]:
                    nome = jogo['clean_game'].title()
                    generos = ', '.join(sorted(set(jogo['genres'].split(';')))) if isinstance(jogo['genres'], str) else "N/A"
                    preco = f"${jogo['price']:.2f}"
                    probabilidade = min(max(jogo['rf_prob'], 0.0), 0.9999)
                    prob_formatada = f"{probabilidade * 100:.2f}%"

                    st.markdown(f"""
                        ### 🎮 {nome}
                        **Gêneros:** `{generos}`  
                        **Preço:** 💲 {preco}  
                        **Probabilidade de Interesse:** 🔮 {prob_formatada}
                        ---
                    """)
    else:
        st.warning(f"Nenhum jogo encontrado com '{search_text}'.")
        if sugestoes:
            st.markdown("👀 Talvez você quis dizer:")
            for s in sugestoes[:5]:
                st.markdown(f"- {s}")
else:
    st.info("Insira o nome (ou parte do nome) de um jogo para ver as recomendações.")
