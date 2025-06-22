# Old Models

Essa seção armazena todo o progresos de desenvolvimento do sistema de recomendação, inicialmente o sistema foi treinado com 3 tipos de modelo:

- Regressão
- Clusterização
- Classificação

O modelos que apresentou melhor resultado foi o `random forest` que é um modelo de classificação que utiliza diversas árvores de decisão para tomar o melhor caminho, tendo uma acurácia de cerca de 88%.

O objetivo final do projeto é utilizar um sistema de recomendão que usa `TF-IDF + cosine similarity` para recomendar os possíveis N jogos, e o `random forest` como um validador com base na probabilidade do usuário gostar do jogo (por exemplo, se ele jogaria mais de 10 horas)

## Estrutura de Arquivos

```bash
old_models/
|
|--- front.py # Primera codificação de uma plataforma web usando streamlit
|--- main.py # arquivo principal de EDA e junção dos datasets
|--- README.md # este arquivo
|--- recomender_interface.py # Sistema de recomendação IF-TDF + consine similarity com tabualação
|--- recommender.py # Primeiro sistema de recomendação IF-TDF + consine similarity via terminal
```