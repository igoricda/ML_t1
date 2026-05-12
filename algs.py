# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error

# Importando os Regressores
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor

# CARREGAMENTO E PREPARAÇÃO DOS DADOS

print("Carregando e preparando dados...")
df_treino = pd.read_csv('./split/treino.csv')
df_teste = pd.read_csv('./split/teste.csv')
df_val = pd.read_csv('./split/validacao.csv')

x_treino = df_treino.drop(columns=['overall'])
y_treino = df_treino['overall']
x_teste = df_teste.drop(columns=['overall'])
y_teste = df_teste['overall']
x_val = df_val.drop(columns=['overall'])
y_val = df_val['overall']

# Normalização
scaler = StandardScaler()
x_treino = scaler.fit_transform(x_treino)
x_teste = scaler.transform(x_teste)
x_val = scaler.transform(x_val)

# Preparação do PredefinedSplit (Treino = -1, Validação = 0)
x_combined = np.vstack((x_treino, x_val))
y_combined = np.hstack((y_treino, y_val))
test_fold = np.zeros(x_combined.shape[0])
test_fold[:len(x_treino)] = -1  
ps = PredefinedSplit(test_fold)

# CONFIGURAÇÃO DOS MODELOS E HIPERPARÂMETROS


modelos_config = {
    "KNN": (
        KNeighborsRegressor(), 
        {
            'n_neighbors': [i for i in range(1, 31)], 
            'weights': ['uniform', 'distance']
        }
    ),
    "Árvore de Decisão": (
        DecisionTreeRegressor(random_state=42), 
        {
            'criterion': ['squared_error', 'absolute_error'], 
            'splitter': ['best', 'random'], 
            # Gera: [None, 5, 10, 15, 20]
            'max_depth': [None] + [i for i in range(5, 25, 5)], 
            # Gera: [2, 10, 20, 30]
            'min_samples_split': [2] + [i for i in range(10, 35, 10)], 
            # Gera: [1, 5, 10, 15]
            'min_samples_leaf': [1] + [i for i in range(5, 20, 5)]
        }
    ),
    "SVM": (
        SVR(), 
        {
            # Gera potências de base 10: [0.1, 1, 10]
            'C': [10**i for i in range(-1, 2)], 
            'kernel': ['linear', 'poly', 'rbf']
        }
    ),
    "MLP": (

        MLPRegressor(random_state=42, early_stopping=True, n_iter_no_change=15, tol=1e-3), 
        {
            
            'max_iter': [500, 1000, 1500], 
            
            # Taxas de aprendizado (0.01 ou 0.001)
            'learning_rate_init': [10**(-i) for i in range(2, 4)], 
            
            'hidden_layer_sizes': [(64,), (128,), (64, 32)], 
            
            'activation': ['relu'], 
            
            'batch_size': [64, 128],

            #Quantidade de camadas (1, 2, 3)
        }
    ),
    "Random Forest": (
        RandomForestRegressor(random_state=42), 
        {
            # Gera: [50, 100, 150, 200]
            'n_estimators': [i * 50 for i in range(1, 5)], 
            'criterion': ['squared_error', 'absolute_error'], 
            # Gera: [None, 10, 20]
            'max_depth': [None] + [i * 10 for i in range(1, 3)], 
            # Gera: [2, 10, 20]
            'min_samples_split': [2] + [i * 10 for i in range(1, 3)], 
            # Gera: [1, 5, 10]
            'min_samples_leaf': [1] + [i * 5 for i in range(1, 3)]
        }
    ),
    "Bagging": (
        BaggingRegressor(random_state=42), 
        {
            # [10, 50, 100, 150]
            'n_estimators': [10] + [i * 50 for i in range(1, 4)], 
            # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            'max_samples': [i / 10 for i in range(5, 11)],
            #estimadores
        }
    ),
    "Boosting": (
        GradientBoostingRegressor(random_state=42), 
        {
            # [50, 100, 150, 200]
            'n_estimators': [i * 50 for i in range(1, 5)], 
            #  [0.1, 0.01, 0.2]
            'learning_rate': [10**(-i) for i in range(1, 3)] + [0.2],
            #estimators - knn, mlp
        }
    )
}

#

# SISTEMA DE CHECKPOINT E EXECUÇÃO

arquivo_resultados = 'resultados_modelos.csv'
modelos_concluidos = []

print("\n" + "="*50)
# Verifica se o arquivo já existe para recuperar o progresso
if os.path.exists(arquivo_resultados):
    df_salvo = pd.read_csv(arquivo_resultados)
    modelos_concluidos = df_salvo['Modelo'].tolist()
    print(f"Progresso recuperado! Modelos já concluídos: {modelos_concluidos}")
else:
    # Se não existe, cria o arquivo com o cabeçalho
    print("Iniciando bateria de testes do zero...")
    pd.DataFrame(columns=['Modelo', 'Melhores_Parametros', 'MAE_Teste']).to_csv(arquivo_resultados, index=False)

print("="*50)

# Loop passando por cada modelo configurado
for nome_modelo, (estimador, param_grid) in modelos_config.items():
    
    # Se o modelo já estiver no CSV, pula para o próximo
    if nome_modelo in modelos_concluidos:
        print(f"Pulando {nome_modelo} (já testado anteriormente).")
        continue

    print(f"\nTreinando {nome_modelo}...")
    
    if param_grid:
        grid = GridSearchCV(estimador, param_grid, cv=ps, n_jobs=-1, scoring='neg_mean_absolute_error')
        grid.fit(x_combined, y_combined)
        melhor_modelo = grid.best_estimator_
        melhores_params = str(grid.best_params_)
    

    # Avaliação no conjunto de Teste
    opiniao = melhor_modelo.predict(x_teste)
    mae_teste = mean_absolute_error(y_teste, opiniao)
    
    print(f"{nome_modelo} concluído! MAE: {mae_teste:.4f} | Params: {melhores_params}")

    # SALVAR PROGRESSO
    novo_resultado = pd.DataFrame([{'Modelo': nome_modelo, 'Melhores_Parametros': melhores_params, 'MAE_Teste': mae_teste}])
    novo_resultado.to_csv(arquivo_resultados, mode='a', header=False, index=False)

# RESULTADO FINAL

print("\n" + "="*50)
print("TODOS OS TESTES FORAM CONCLUÍDOS COM SUCESSO!")
print("="*50)

# Ler o CSV completo e mostrar o melhor modelo
df_final = pd.read_csv(arquivo_resultados)
melhor_resultado = df_final.loc[df_final['MAE_Teste'].idxmin()]

print(f"\nO MELHOR MODELO FOI: {melhor_resultado['Modelo']}")
print(f"MAE Alcançado: {melhor_resultado['MAE_Teste']:.4f}")
print(f"Hiperparâmetros: {melhor_resultado['Melhores_Parametros']}")
print(f"\nOs dados completos estão salvos no arquivo: '{arquivo_resultados}'")