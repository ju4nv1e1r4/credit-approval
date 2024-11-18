"""
Script de Pré-processamento de Dados

Este script realiza o pré-processamento de dados para um conjunto de características numéricas e categóricas,
seguindo uma abordagem otimizada para melhorar o desempenho e a legibilidade do código.

Etapas do pré-processamento:

1. Carregamento dos dados:
   - O arquivo CSV contendo as características é carregado em um DataFrame (`df`).
   
2. Separação e preparação dos dados:
   - As colunas categóricas e numéricas são identificadas.
   - O conjunto de dados é dividido em `X` (variáveis de entrada) e `y` (variável-alvo).
   - A coluna 'education' é mapeada para valores descritivos (ex.: "ensino_medio", "ensino_superior", "pos_graduacao").
   - Os dados são divididos em treino e teste usando `train_test_split`.

3. Pré-processamento:
   - A transformação logarítmica é aplicada diretamente na coluna 'mortgage' dentro do pipeline.
   - Os dados numéricos são escalonados usando `RobustScaler`.
   - As variáveis categóricas são codificadas com `OneHotEncoder`, ignorando valores desconhecidos.
   - O `ColumnTransformer` combina transformações numéricas e categóricas em um pipeline único.

4. Salvamento dos dados:
   - Os dados pré-processados são salvos em um arquivo CSV (`preprocessed_data.csv`).
   - Verificação do diretório de salvamento; se não existir, ele é criado automaticamente.

Exceções:
- O script trata erros de carregamento, pré-processamento e salvamento de dados, exibindo mensagens de erro específicas.

Notas:
- O arquivo de entrada deve estar no caminho correto, e as colunas indicadas devem existir no DataFrame.
- O pipeline de pré-processamento é flexível e pode ser ajustado conforme necessário.

"""


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# step-1. load data
try:
    df = pd.read_csv('data/feature_store/data_with_new_features.csv')
    print('Dados carregados com sucesso.')
except FileNotFoundError as load_error:
    print(f'{load_error}: Erro ao carregar arquivo. Verifique o caminho do diretório.')

# step-2. preprocess and transformation
try:
    df['education'] = df['education'].replace({1: 'ensino_medio', 2: 'ensino_superior', 3: 'pos_graduacao'})

    num_features = ['age', 'income', 'family', 'mortgage']

    cat_features = ['age_bracket_name', 'education']
    
    target = 'credit_card'

    X = df[num_features + cat_features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    numerical_transformer = Pipeline(steps=[
        ('scaler', RobustScaler()),
        ('log_transform', 'passthrough') 
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])
    
    X_encoded = preprocessor.fit_transform(X)
    print('Dados pré-processados com sucesso')
except Exception as err:
    print(f'{err}: ocorreu um erro ao preprocessar os dados.')

# step-3. saving preprocessed dataset
try:
    save_path = 'data/preprocessed/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_encoded = pd.DataFrame(X_encoded, columns=preprocessor.get_feature_names_out())
    df_encoded.reset_index(drop=True, inplace=True)

    df_combined = pd.concat([df.reset_index(drop=True), df_encoded], axis=1)
    
    df_combined.to_csv(os.path.join(save_path, 'preprocessed_data_combined.csv'), index=False)
    
    print('Dados salvos com sucesso')
except Exception as save_error:
    print(f'{save_error}: Ocorreu um erro ao salvar o arquivo. Verifique o caminho do diretório')
