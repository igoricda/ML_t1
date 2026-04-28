import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Carregar o dataset original
df = pd.read_csv('fifa_previsao_ajustado.csv')

# 2. Realizar a divisão (80% treino, 10% validação, 10% teste)
# Primeiro, separamos o Treino (80%) do resto (20%)
df_train, df_temp = train_test_split(df, test_size=0.50, random_state=42)

# Depois, dividimos o resto ao meio para Validação (10%) e Teste (10%)
df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42)

# 3. Criar os arquivos .csv
# index=False evita que o Pandas crie uma coluna extra com os números das linhas
df_train.to_csv('treino.csv', index=False)
df_val.to_csv('validacao.csv', index=False)
df_test.to_csv('teste.csv', index=False)

print("Arquivos criados com sucesso: treino.csv, validacao.csv e teste.csv")