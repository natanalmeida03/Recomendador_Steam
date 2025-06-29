import streamlit as st
import pandas as pd
from recommender import recomendar_jogos 

# ===========================
# Configuração da Página
# ===========================
st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="wide"
)

# ===========================
# Custom CSS
# ===========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

    body {
        font-family: 'Poppins', sans-serif;
        background-color: #1a0f3d; 
        color: #e0e0e0; 
    }

    .stApp {
        background-color: #1a0f3d; 
    }

    /* Title styling */
    h1 {
        font-family: 'Montserrat', sans-serif;
        color: #ff33bb; 
        text-align: center;
        margin-bottom: 30px;
        font-size: 3.5em;
        font-weight: 700;
        text-shadow: 0 0 15px rgba(255, 51, 187, 0.6); 
    }

    /* Subheader styling */
    h2 {
        font-family: 'Montserrat', sans-serif;
        color: #00e6e6; 
        border-bottom: 2px solid #3366ff; 
        padding-bottom: 10px;
        margin-top: 40px;
        font-weight: 600;
    }

    /* Markdown text styling 
    .stMarkdown p {
        font-size: 1.1em;
        line-height: 1.6;
    }

    /* Form and input styling */
    .stTextInput label, .stSlider label {
        color: #aaffff;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
    }

    .stTextInput input[type="text"] {
        background-color: #2b1f5e;
        color: #e0e0e0;
        border: 1px solid #3366ff; 
        border-radius: 10px;
        padding: 12px 18px;
        font-size: 1.1em;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.3);
    }

    .stSlider .stSlider-value {
        color: #00ff99; 
        font-weight: bold;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #ff33bb, #8833ff); 
        color: white;
        border: none;
        border-radius: 10px; 
        padding: 12px 25px;
        font-size: 1.2em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.4);
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.5);
        opacity: 0.9;
    }

    /* Recommendation card styling*/
    .game-card {
        background-color: #2b1f5e; 
        border-radius: 15px; 
        padding: 25px; 
        margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4); 
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%; 
        border: 1px solid #4a3a8a;
    }

    .game-card:hover {
        transform: translateY(-7px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.6);
    }

    .game-card h3 {
        color: #ff8800; 
        border-bottom: none;
        padding-bottom: 0;
        margin-top: 0;
        margin-bottom: 15px; 
        font-family: 'Montserrat', sans-serif;
        font-size: 1.6em;
        font-weight: 700;
        word-wrap: break-word;
        text-shadow: 0 0 8px rgba(255, 136, 0, 0.4);
    }

    .game-card .info-section {
        margin-bottom: 15px; 
    }

    .game-card strong {
        color: #e0e0e0; 
        font-weight: 600; 
    }

    .game-card .genres-container {
        display: flex;
        flex-wrap: wrap;
        gap: 5px; 
        margin-top: 5px;
    }

    .game-card .genres-tag { 
        background-color: #4a3a8a; 
        color: #aaffff;
        border-radius: 5px;
        padding: 4px 10px;
        font-size: 0.85em;
        border: 1px solid #6a5a9a;
        white-space: nowrap; 
    }

    .game-card .price, .game-card .probability {
        font-weight: bold;
        font-size: 1.1em;
        display: block; 
        margin-top: 5px; 
    }

    .game-card .price-value {
        color: #00ff99; 
    }

    .game-card .probability-value {
        color: #ffcc00; 
    }

    .game-card .card-footer {
        border-top: 1px solid #4a3a8a; 
        padding-top: 15px;
        margin-top: auto; 
    }
            
    .game-card .like-button, .game-card .dislike-button {
        background-color: #E57A00;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 15px;
        font-size: 1em;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.3s ease;
        margin-right: 10px; 
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0.3);
    }
            
    .game-card .like-button:hover, .game-card .dislike-button:hover {
        background-color: #ff8800; 
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.4);
    }

    hr {
        border-top: 1px solid #4a3a8a; 
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Streamlit specific message styling (warning, info) */
    .st-emotion-cache-r421ms { 
        background-color: rgba(255, 51, 187, 0.2);
        color: #ff33bb; /* Pink text */
        border-left: 5px solid #ff33bb;
        padding: 15px;
        border-radius: 10px;
        font-weight: 600;
    }

    .st-emotion-cache-16cq8pd { 
        background-color: rgba(51, 102, 255, 0.2); 
        color: #3366ff; /* Blue text */
        border-left: 5px solid #3366ff;
        padding: 15px;
        border-radius: 10px;
        font-weight: 600;
    }

    /* Style for suggestion list */
    .suggestion-list {
        background-color: #2b1f5e;
        border: 1px solid #3366ff;
        border-radius: 10px;
        padding: 15px 20px;
        margin-top: 20px;
        list-style-type: disc; 
        padding-left: 30px;
    }

    .suggestion-list li {
        color: #aaffff;
        margin-bottom: 8px;
        font-family: 'Poppins', sans-serif;
        font-size: 1.05em;
    }

</style>
""", unsafe_allow_html=True)

# ===========================
# Cabeçalho
# ===========================
st.title("Steam Game Recommender")
st.markdown(
    """
    Obtenha recomendações personalizadas de jogos da Steam com base em **conteúdo textual (TF-IDF + Cosine Similarity)** e um modelo **Random Forest** que estima sua probabilidade de gostar de cada jogo.
    """
)

# ===========================
# Formulário
# ===========================
with st.form("recommender_form"):
    search_text = st.text_input("Digite parte do nome do jogo:", "")
    top_n = st.slider("Número de recomendações:", 1, 10, 5)
    submit = st.form_submit_button("Recomendar")

# ===========================
# Recomendação
# ===========================
if submit and search_text.strip():
    st.subheader("Resultados")

    recomendacoes, sugestoes = recomendar_jogos(search_text.lower().strip(), top_n)

    if recomendacoes is not None and not recomendacoes.empty:
        num_columns = 2
        effective_num_columns = max(1, num_columns)
        rows = [recomendacoes.iloc[i:i+effective_num_columns] for i in range(0, len(recomendacoes), effective_num_columns)]

        for row_slice in rows:
            cols = st.columns(effective_num_columns)
            for i, (_, jogo) in enumerate(row_slice.iterrows()):
                with cols[i]:
                    nome = jogo['clean_game'].title()
                    generos_list = sorted(set(jogo['genres'].split(';'))) if isinstance(jogo['genres'], str) else []
                    
                    generos_html = ''.join([f'<span class="genres-tag">{g.strip()}</span>' for g in generos_list if g.strip()])

                    preco = f"${jogo['price']:.2f}"
                    probabilidade = min(max(jogo['rf_prob'], 0.0), 0.9999)
                    prob_formatada = f"{probabilidade * 100:.2f}%"

                    card_full_html = f"""
                    <div class="game-card">
                        <h3>🎮 {nome}</h3>
                        <div class="info-section">
                            <strong>Gêneros:</strong>
                            <div class="genres-container">{generos_html if generos_html else "N/A"}</div>
                        </div>
                        <div class="card-footer">
                            <p class="price"><strong>Preço:</strong> <span class="price-value">{preco}</span></p>
                            <p class="probability"><strong>Probabilidade de Interesse:</strong> <span class="probability-value">{prob_formatada}</span></p>
                            <button class="like-button" onclick="alert('Você curtiu {nome}!')"><strong>Curtir</strong></button>
                            <button class="dislike-button" onclick="alert('Você não curtiu {nome}.')"><strong>Não Curtir</strong></button>
                        </div>
                    </div>
                    """
                    st.markdown(card_full_html, unsafe_allow_html=True)
    else:
        st.warning(f"Nenhum jogo encontrado com '{search_text}'.")
        if sugestoes:
            st.markdown("Talvez você quis dizer:")
            st.markdown('<ul class="suggestion-list">', unsafe_allow_html=True)
            for s in sugestoes[:5]: 
                st.markdown(f"<li>{s}</li>", unsafe_allow_html=True)
            st.markdown('</ul>', unsafe_allow_html=True)
else:
    st.info("Insira o nome (ou parte do nome) de um jogo para ver as recomendações.")