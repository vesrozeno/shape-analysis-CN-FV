import os
import pandas as pd
import matplotlib.pyplot as plt

# Caminho da pasta base que contém os arquivos CSV
results_directory = "results/otolith/"

# Parâmetros de filtro específicos
selected_n_modes = 20
selected_thre_inc = 35

# Listar todos os arquivos CSV no diretório de resultados
csv_files = [f for f in os.listdir(results_directory) if f.endswith('.csv')]

# Dicionário para armazenar as combinações de (d, f, c) e as acurácias de cada arquivo
results_dict = {}

# Iterar sobre cada arquivo CSV e coletar os dados
for csv_file in csv_files:
    # Extrair o nome do arquivo sem a extensão .csv
    file_name = os.path.splitext(csv_file)[0]
    
    file_path = os.path.join(results_directory, csv_file)
    df = pd.read_csv(file_path)
    
    # Filtrar com base em n_modes e thre_inc específicos
    df_filtered = df[(df['n_modes'] == selected_n_modes) & (df['thre_inc'] == selected_thre_inc)]
    
    # Iterar sobre as linhas filtradas e adicionar os dados ao dicionário
    for _, row in df_filtered.iterrows():
        d = row['d']
        f = row['f']
        c = row['c']
        lda_acc = row['lda_acc']
        svm_acc = row['svm_acc']
        
        # Chave baseada na combinação (d, f, c)
        key = (d, f, c)
        
        # Se a chave não existir no dicionário, inicializa uma nova lista
        if key not in results_dict:
            results_dict[key] = {}
        
        # Adiciona as acurácias LDA e SVM para o arquivo CSV atual, usando o nome do arquivo como chave
        results_dict[key][f"{file_name}_lda"] = lda_acc
        results_dict[key][f"{file_name}_svm"] = svm_acc

# Criar um DataFrame para organizar a tabela
rows = []
for key, acc_dict in results_dict.items():
    d, f, c = key
    row = {'d': d, 'f': f, 'c': c}
    
    # Adicionar as acurácias com base no nome do arquivo
    row.update(acc_dict)
    
    # Adicionar a linha à lista de linhas
    rows.append(row)

# Converter a lista de linhas em um DataFrame
results_df = pd.DataFrame(rows)

# Organizar colunas, mantendo 'd', 'f', 'c' primeiro e os arquivos em pares lda/svm
ordered_columns = ['d', 'f', 'c'] + sorted([col for col in results_df.columns if col not in ['d', 'f', 'c']])
results_df = results_df[ordered_columns]

# Exibir a tabela
print("\nTabela de acurácias do LDA e SVM para n_modes =", selected_n_modes, "e thre_inc =", selected_thre_inc)
print(results_df)
results_df.to_csv("/mnt/d/parameters/oto3520_lda_accuracies_table.csv", index=False)
