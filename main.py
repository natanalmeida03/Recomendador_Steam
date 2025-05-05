import numpy as np
import pandas as pd
import re

# Funções de tratamento
def clean_name(name: str):
    if pd.isna(name):
        return name
    new_name = re.sub(r"\s{2,}|\t", " ", name).strip().lower()
    new_name = re.sub(r"[^\w\s\d]+", "", new_name)
    return new_name

def get_median(range_str: str):
    if pd.isna(range_str):
        return None
    if '-' in range_str:
        lower, upper = range_str.split('-')
        return (int(upper) + int(lower)) / 2
    return None

# Carregar dataset de jogos
df_games = pd.read_csv('./data/steam.csv')

# Tratamento do dataset de jogos
df_games['release_date'] = pd.to_datetime(df_games['release_date'], format='%Y-%m-%d', errors='coerce')
df_games['release_year'] = df_games['release_date'].dt.year
df_games['clean_game'] = df_games['name'].apply(clean_name)
df_games['owners'] = df_games['owners'].apply(get_median)
df_games['owners'] = pd.to_numeric(df_games['owners'], errors='coerce', downcast='integer')

# Tratamento de valores nulos
df_games = df_games.dropna(subset=['owners', 'price', 'positive_ratings', 'negative_ratings'])

# Excluir colunas desnecessárias
df_games.drop(columns=['appid', 'english', 'developer', 'publisher'], inplace=True)

# Carregar e tratar dataset de usuários
df_users = pd.read_csv('./data/steam-200k.csv')
df_users.columns = df_users.columns.str.strip()  # Remove espaços dos nomes das colunas

# Renomear coluna 'value' para 'value' (caso haja espaços)
df_users.rename(columns=lambda x: x.strip(), inplace=True)

df_users['clean_game'] = df_users['game'].apply(clean_name)
df_users = df_users.loc[df_users['action'] == 'play', :].copy()
df_users.drop(columns=['action', 'unused'], inplace=True)
df_users = df_users.dropna(subset=['value'])
df_users = df_users[df_users['value'] >= 0]

# Merge
df = pd.merge(df_users, df_games, how='inner', on=['clean_game'])
df.drop(columns=['name'], inplace=True)

# Converter colunas categóricas para tipo 'category'
categorical_cols = ['platforms', 'categories', 'genres', 'steamspy_tags']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Gravar datasets limpos
df.to_csv('./data/steam_project.csv', index=False)