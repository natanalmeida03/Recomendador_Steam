import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Configurações iniciais
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')
sns.set_palette("husl")

# ==============================================
# 1. FUNÇÕES DE TRATAMENTO DE DADOS APRIMORADAS
# ==============================================

def clean_name(name: str) -> str:
    """Limpeza avançada de nomes de jogos"""
    if pd.isna(name):
        return name
    # Remover caracteres especiais e normalizar espaços
    new_name = re.sub(r"\s{2,}|\t", " ", name).strip().lower()
    new_name = re.sub(r"[^\w\s\dáéíóúâêîôûãõç]", "", new_name)
    return new_name

def process_owners(range_str: str) -> float:
    """Processamento robusto da coluna de owners"""
    if pd.isna(range_str):
        return np.nan
    try:
        if '-' in range_str:
            lower, upper = map(lambda x: x.strip(), range_str.split('-'))
            return (int(upper.replace(',','')) + int(lower.replace(',',''))) / 2
        return float(range_str.replace(',',''))
    except:
        return np.nan

def treat_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Tratamento de outliers usando método IQR"""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df

def expand_categorical_columns(df: pd.DataFrame, column: str, sep: str = ';') -> pd.DataFrame:
    """Expande colunas categóricas multivaloradas em one-hot encoding"""
    expanded = df[column].str.split(sep).explode().str.get_dummies().groupby(level=0).sum()
    return pd.concat([df, expanded.add_prefix(f'{column}_')], axis=1)

# ==============================================
# 2. CARREGAMENTO E PROCESSAMENTO DE DADOS
# ==============================================

# Carregar e tratar dataset de jogos
df_games = pd.read_csv('./data/steam.csv')

# Salvar uma cópia do dataset original antes de qualquer processamento
df_games_original = df_games.copy()

# Limpeza e transformações
df_games['release_date'] = pd.to_datetime(df_games['release_date'], errors='coerce')
df_games['release_year'] = df_games['release_date'].dt.year
df_games['clean_name'] = df_games['name'].apply(clean_name)
df_games['owners'] = df_games['owners'].apply(process_owners)
df_games['price'] = pd.to_numeric(df_games['price'], errors='coerce')

# Engenharia de features para jogos
df_games = expand_categorical_columns(df_games, 'genres')
df_games = expand_categorical_columns(df_games, 'steamspy_tags')

# Tratamento de outliers numéricos
numeric_cols = ['price', 'positive_ratings', 'negative_ratings']
for col in numeric_cols:
    # Salvar valores antes do tratamento
    original_values = df_games_original[col]
    
    # Tratar outliers
    df_games = treat_outliers(df_games, col)
    
    # Calcular diferença
    treated_values = df_games[col]
    difference = (original_values - treated_values).abs()
    
    # Imprimir resumo da diferença
    print(f"\nDiferença para a coluna '{col}':")
    print(f"Máxima diferença: {difference.max()}")
    print(f"Média diferença: {difference.mean()}")
    print(f"Total de valores alterados: {(difference > 0).sum()}")

# Carregar e tratar dataset de usuários
df_users = pd.read_csv(
    './data/steam-200k.csv',
    header=None,
    names=['user_id', 'game', 'action', 'value', 'zero'],
    engine='python'
)

# Salvar uma cópia do dataset de usuários antes de qualquer processamento
df_users_original = df_users.copy()

# Processamento avançado de usuários
df_users['clean_game'] = df_users['game'].apply(clean_name)
df_users = df_users.query("action == 'play' and value >= 0").copy()

# Salvar valores antes do tratamento
original_values_users = df_users['value']

# Tratar outliers no dataset de usuários
df_users = treat_outliers(df_users, 'value')

# Calcular diferença
treated_values_users = df_users['value']
difference_users = (original_values_users - treated_values_users).abs()

# Imprimir resumo da diferença para usuários
print(f"\nDiferença para a coluna 'value' (playtime):")
print(f"Máxima diferença: {difference_users.max()}")
print(f"Média diferença: {difference_users.mean()}")
print(f"Total de valores alterados: {(difference_users > 0).sum()}")

# Renomear a coluna tratada
df_users.rename(columns={'value': 'playtime_hours'}, inplace=True)

# print(df_users.columns)
# print(df_games.columns)

# ==============================================
# 3. ANÁLISE EXPLORATÓRIA INTEGRADA
# ==============================================

def plot_feature_distribution(df: pd.DataFrame, column: str, log_scale: bool = False):
    """Visualização comparativa de distribuições antes/depois do tratamento"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribuição Original - {column}')
    if log_scale:
        plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribuição Tratada - {column}')
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

# Análise de preços
plot_feature_distribution(df_games, 'price', log_scale=True)

# Análise de playtime
plot_feature_distribution(df_users, 'playtime_hours', log_scale=True)

# ==============================================
# 4. MERGE E FEATURE ENGINEERING
# ==============================================

# Merge dos datasets
df_merged = pd.merge(
    df_users,
    df_games,
    how='inner',
    left_on='clean_game',
    right_on='clean_name',
    suffixes=('_user', '_game')
)

# Engenharia de features combinadas
df_merged['user_engagement'] = df_merged['playtime_hours'] * df_merged['positive_ratings']
df_merged['price_per_hour'] = df_merged['price'] / df_merged['playtime_hours'].replace(0, 0.1)

# Normalização de features
scaler = MinMaxScaler()
df_merged[['playtime_norm', 'price_norm']] = scaler.fit_transform(
    df_merged[['playtime_hours', 'price']]
)

# ==============================================
# 5. ANÁLISE DE CORRELAÇÃO
# ==============================================

def plot_correlation_matrix(df: pd.DataFrame, columns: list):
    """Visualização de matriz de correlação"""
    plt.figure(figsize=(15, 10))
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação')
    plt.show()

# Selecionar features numéricas para análise
numeric_features = [
    'playtime_hours',
    'price',
    'positive_ratings',
    'negative_ratings',
    'user_engagement',
    'price_per_hour'
]

plot_correlation_matrix(df_merged, numeric_features)

# ==============================================
# 6. ANÁLISE TEMPORAL
# ==============================================

def plot_temporal_trends(df: pd.DataFrame):
    """Análise de tendências temporais"""
    plt.figure(figsize=(14, 6))
    
    # Tendência de lançamentos por ano
    plt.subplot(1, 2, 1)
    df[df['release_year'] > 1990]['release_year'].value_counts().sort_index().plot()
    plt.title('Lançamentos por Ano')
    plt.xlabel('Ano')
    plt.ylabel('Número de Jogos')
    
    # Tendência de preço médio por ano
    plt.subplot(1, 2, 2)
    df.groupby('release_year')['price'].mean().plot()
    plt.title('Preço Médio por Ano de Lançamento')
    plt.xlabel('Ano')
    plt.ylabel('Preço Médio (USD)')
    
    plt.tight_layout()
    plt.show()

plot_temporal_trends(df_games)

# ==============================================
# 7. SALVAMENTO E EXPORTAÇÃO
# ==============================================

# Selecionar colunas relevantes
final_columns = [
    'user_id', 'clean_game', 'playtime_hours', 'price',
    'genres', 'steamspy_tags', 'release_year', 'user_engagement',
    'price_per_hour', 'playtime_norm', 'price_norm'
]

# Exportar dataset final
df_merged[final_columns].to_csv('./data/steam_enhanced.csv', index=False)

print("Processamento concluído com sucesso!")