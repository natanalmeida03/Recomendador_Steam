#  RECOMENDADOR DE JOGOS STEAM

Este sistema foi modelado durante e disciplina de `Aprendizado de Máquina` da Universidade de Brasília (UnB), ministrada pelo professor [Sergio Antônio Andrade de Freitas](https://github.com/sergioaafreitas)

[Repositório da disciplina](https://github.com/sergioaafreitas/CAM)

Sistema inteligente de recomendação de jogos da Steam utilizando técnicas de **NLP (TF-IDF + Cosine Similarity)** combinadas com **aprendizado supervisionado (Random Forest)** para prever jogos com maior chance de agradar ao usuário.

## Funcionalidades

- **Busca de Jogos:** Digite parte do nome de um jogo e receba recomendações personalizadas.
- **TF-IDF + Cosine Similarity:** Análise semântica baseada em gêneros, categorias e tags dos jogos.
- **Random Forest:** Modelo supervisionado que estima a probabilidade de o usuário gostar do jogo.
- **Interface Interativa:** Visualização em grade com informações organizadas via **Streamlit**.
- **Gráficos e Métricas:** Avaliação de modelos de regressão, classificação e clusterização com métricas como R², MAE, F1 Score e Silhouette Score.

## Tecnologias Utilizadas

- **Python**
- **Pandas / NumPy / Scikit-learn**
- **TF-IDF (TfidfVectorizer)**
- **Cosine Similarity**
- **Random Forest Classifier**
- **KMeans + PCA**
- **Streamlit** (interface gráfica)

## Lógica da Recomendação

1. Pré-processamento dos dados:
   - Combinação de `genres` e `tags` em um campo textual.
   - Aplicação de TF-IDF com bigramas.
   - Normalização de features numéricas.

2. Cálculo da similaridade:
   - Cálculo da distância do jogo consultado com os demais.
   - Ordenação dos jogos por **Cosine Similarity**.

3. Previsão de interesse:
   - Utilização do modelo **Random Forest** treinado com `playtime_hours > 10` como variável alvo.
   - Cada jogo recomendado traz sua **probabilidade estimada** de agradar ao jogador.

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
