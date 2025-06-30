#  RECOMENDADOR DE JOGOS STEAM

Este sistema foi modelado durante e disciplina de `Aprendizado de Máquina` da Universidade de Brasília (UnB), ministrada pelo professor [Sergio Antônio Andrade de Freitas](https://github.com/sergioaafreitas)

[Repositório da disciplina](https://github.com/sergioaafreitas/CAM)

Este projeto implementa um sistema de recomendação de jogos para a plataforma Steam, utilizando uma abordagem híbrida que combina similaridade de conteúdo (TF-IDF + Similaridade de Cosseno), agrupamento de jogos (K-Means) e um modelo de Machine Learning (Random Forest Regressor) para prever o engajamento do usuário.

[Acesse online](https://recomendador-steam-cam.streamlit.app/)

## Arquitetura do Sistema

O sistema é composto por três módulos principais:

1.  **`recommender.py`**: Este é o coração do sistema de recomendação. Ele é responsável por:
    * Carregar e pré-processar os dados de jogos Steam.
    * Gerar representações textuais (gêneros, tags) usando TF-IDF.
    * Realizar o agrupamento de jogos (K-Means) com base em suas características numéricas (`price`, `user_engagement`, `price_per_hour`, `playtime_norm`, `price_norm`). Cada jogo é associado a um cluster.
    * Construir uma matriz combinada de features textuais e numéricas para calcular a similaridade de cosseno.
    * Treinar o modelo `RandomForestRegressor` para prever o `user_engagement` (engajamento do usuário), utilizando as características do jogo e o cluster ao qual ele pertence como features preditoras.
    * A função `recomendar_jogos` recebe um nome de jogo, encontra jogos similares usando TF-IDF + Similaridade de Cosseno, refina essa lista aplicando o modelo de regressão para prever o engajamento de cada jogo candidato, e retorna as recomendações ranqueadas pelo engajamento predito.

2.  **`model.py` (ou `random_forest.py`)**: Este script é dedicado ao ajuste fino (tuning) do modelo `RandomForestRegressor`. Ele realiza:
    * Carga e pré-processamento de dados de forma consistente com `recommender.py`.
    * Treinamento do K-Means para atribuir clusters aos jogos.
    * Definição da variável alvo para regressão: `user_engagement` (aplicando `log1p` para tratar assimetria).
    * Utilização de `GridSearchCV` para encontrar os melhores hiperparâmetros para o `RandomForestRegressor`, usando métricas de regressão como RMSE e R2 Score.
    * As features usadas para o Random Forest incluem `price`, `price_per_hour`, `playtime_norm`, `price_norm` e o `game_cluster`.

3.  **`front.py`**: Este arquivo é a interface do usuário do sistema de recomendação. Ele:
    * Recebe a entrada do usuário (nome do jogo).
    * Chama a função `recomendar_jogos` de `recommender.py`.
    * Formata e exibe as recomendações em um formato HTML, incluindo o nome do jogo, gêneros, preço e o **Engajamento Predito (Probabilidade de Interesse)** (normalizado para uma escala de 0-100%).

## Por que usar log no Random Forest?

Variáveis como `user_engagement` ou `playtime_hours` frequentemente exibem uma distribuição altamente assimétrica à direita, com muitos jogos tendo engajamento baixo e uma cauda longa de poucos jogos com engajamento extremamente alto (outliers)

## Tecnologias Utilizadas

- **Python**
- **Pandas / NumPy / Scikit-learn**
- **TF-IDF (TfidfVectorizer)**
- **Cosine Similarity**
- **Random Forest Classifier**
- **KMeans + PCA**
- **Streamlit** (interface gráfica)

## Interface Gráfica

A interface é construída com **Streamlit**, exibindo:

- Campo de busca parcial por nome
- Controle de número de recomendações
- Grid com jogos recomendados
- Informações como gêneros, preço e chance de interesse

## Modelos Avaliados

| Tipo           | Modelos Testados                              | Melhor Resultado |
|----------------|-----------------------------------------------|------------------|
| Classificação  | Random Forest                                 | 89% Accuracy     |
| Regressão      | Linear, KNN, Gradient Boosting                | R² = 0.9999 (GB) |
| Clusterização  | KMeans com PCA                                | Silhouette 0.55  |
| Recomendação   | TF-IDF + Cosine Similarity + Random Forest    | Relevância Visual|

## Como executar

clone o repositorio

```bash
git clone https://github.com/natanalmeida03/Recomendador_Steam.git
```

Entre na pasta
```bash
cd Recomendador_Steam
```
crie um ambiente virtual

```bash
python -m venv venv
```

Ative o ambiente virtual

```bash
# Windows
venv\Scripts\activate
# Linux
source venv/bin/activate
```
Instale as dependências

```bash
pip install -r requirements.txt
```
Execute o arquivo front

```bash
streamlit run front.py
```

## Equipe

    221008580 - Eduardo Belarmino Silva 
    221029220 - Guilherme Davila Rodrigues Carneiro Sampaio 
    211031744 - Júlio César Costa 
    222006169 - Natan Da Cruz Almeida 
    221022408 - Paulo Henrique Lamounier Dantas

## Resultados

- Sistema funcional e responsivo
- Modelos otimizados com validação cruzada
- Interface prática e acessível
- Base pronta para extensões como recomendação por perfil de usuário (Steam API)

## Melhorias Futuras

- [x] Adicionar Random Forest na previsão dos jogos 
- [ ] Integração com API pública da Steam
- [ ] Avaliação por feedback do usuário (Reinforcement Learning)
