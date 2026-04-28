import pandas as pd
import os

# Carregar a base de dados
df = pd.read_csv('fifa_irl_data.csv')

# Filtrar apenas as versões FIFA 24 e FIFA 25
df = df[df['fifa_version'].isin([24, 25])].copy()

# Remover os Goleiros (GK)
df = df[~df['player_positions'].str.contains('GK', na=False)]

# Processar Posições Dinamicamente
df['position_main'] = df['player_positions'].str.split(',').str[0].str.strip()

pos_mapping = {}
current_id = 1

def map_position(pos):
    global current_id
    if pd.isna(pos):
        return None
    if pos not in pos_mapping:
        pos_mapping[pos] = current_id
        current_id += 1
    return pos_mapping[pos]

df['position_num'] = df['position_main'].apply(map_position)



# Dicionário para armazenar as respostas

league_levels = {}
df_ligas = pd.read_csv('relacao_ligas.csv')
dict_ligas = dict(zip(df_ligas['league_id'], df_ligas['league_level']))

df['league_level'] = df['league_id'].map(dict_ligas)
df = df.drop(columns=['league_id'])

#Definir estatísticas para NÃO incluir no target
cols_to_drop = [
    'fifa_version',
    'league_id',
    'club_team_id'
    'fifa_version',
    'short_name',
    'long_name',
    'player_positions',
    'position_main', 
    'potential',
    'value_eur',
    'wage_eur',
    'age_fifa',
    'height_cm',
    'weight_kg',
    'club_name',
    'league_name',
    'club_jersey_number',
    'nationality_name',
    'preferred_foot',
    'weak_foot',
    'skill_moves',
    'international_reputation',
    'body_type',
    'pace',
    'shooting',
    'passing',
    'dribbling',
    'defending',
    'physic',
    'goalkeeping_diving',
    'goalkeeping_handling',
    'goalkeeping_kicking',
    'goalkeeping_positioning',
    'goalkeeping_reflexes',
    'goalkeeping_speed',
    'born_fifa',
    'season',
    'player',
    'Playing Time_Min',
    'Playing Time_Starts',
    'league',
    'team',
    'nation',
    'pos',
    'age_fbref',
    'born_fbref',
    'Per 90 Minutes_PKatt',
    'Per 90 Minutes_CrdY',
    'Per 90 Minutes_CrdR',
    'Per 90 Minutes_G+A',
    'Per 90 Minutes_G-PK',
    'Per 90 Minutes_G+A-PK',
    'Per 90 Minutes_xG+xAG',
    'Per 90 Minutes_npxG',
    'Per 90 Minutes_npxG+xAG',
    'Standard_Dist',
    'Expected_npxG/Sh',
    'Per 90 Minutes_Expected_xA',
    'Per 90 Minutes_Expected_A-xAG',
    'Performance_GA90',
    'Per 90 Minutes_Performance_SoTA',
    'Per 90 Minutes_Performance_Saves',
    'Performance_Save%',
    'Per 90 Minutes_Performance_W',
    'Per 90 Minutes_Performance_D',
    'Per 90 Minutes_Performance_L',
    'Per 90 Minutes_Performance_CS',
    'Performance_CS%',
    'Per 90 Minutes_Penalty Kicks_PKatt',
    'Per 90 Minutes_Penalty Kicks_PKA',
    'Per 90 Minutes_Penalty Kicks_PKsv',
    'Per 90 Minutes_Penalty Kicks_PKm',
    'Penalty Kicks_Save%',
    'name_parts'
]

# Remover as colunas indesejadas
df_final = df.drop(columns=cols_to_drop, errors='ignore').copy()

# Remover jogadores com dados nulos (NaN)
df_final = df_final.dropna()

# Salvar o CSV principal ajustado
df_final.to_csv('fifa_previsao_ajustado.csv', index=False)

# SALVAR A RELAÇÃO DE POSIÇÕES E IDs EM UM NOVO CSV
# Converte o dicionário pos_mapping em uma lista de tuplas e depois em DataFrame
df_posicoes = pd.DataFrame(list(pos_mapping.items()), columns=['posicao_nome', 'posicao_id'])
df_posicoes.to_csv('relacao_posicoes.csv', index=False)

print("Processamento concluído!")
print("Arquivos gerados: 'fifa_previsao_ajustado.csv' e 'relacao_posicoes.csv'")
print(f"Total de posições mapeadas: {len(df_posicoes)}")