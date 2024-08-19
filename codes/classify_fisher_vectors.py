#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para classificar Vetores de Fisher usando SVM e LDA.

Criado em: 21 Março 2024
Autores: Lucas C. Ribas e Vitor Emanuel S. Rozeno
"""

import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import csv
import os

# Variáveis para determinar os nomes de arquivos
fisher_vectors_pkl = "pkl/fisher_vectors_and_targets_portuguese.pkl"  # Arquivo pkl com os Fisher Vectors e rótulos
results_csv = "results/results_portuguese.csv"  # Arquivo CSV para salvar os resultados

def main():
    # Lendo combinações já processadas
    processed_combinations = read_processed_combinations(results_csv)
    print(f"Número de combinações já processadas: {len(processed_combinations)}")

    print("Carregando Fisher Vectors e rótulos...")
    # Carregar os Fisher Vectors e os rótulos
    with open(fisher_vectors_pkl, "rb") as f:
        data = pickle.load(f)
    fisher_vectors_dict = data["fisher_vectors"]
    targets = data["targets"]

    print("Fisher Vectors e rótulos carregados com sucesso.\n")
    print(f"Length of fisher_vectors: {len(fisher_vectors_dict)}")
    print(f"Length of targets: {len(targets)}")

    # Executar a classificação para cada combinação de parâmetros com uma barra de progresso
    with tqdm(total=len(fisher_vectors_dict), desc="Classificando combinações de parâmetros") as pbar:
        for params, fisher_vectors in fisher_vectors_dict.items():
            d_ctrl, f_ctrl, c_ctrl, k, N = params

            # Verificando se a combinação já foi processada
            if (d_ctrl, f_ctrl, c_ctrl, k, N) in processed_combinations:
                pbar.update(1)  # Atualiza a barra de progresso mesmo para combinações já processadas
                continue

            # Executar a validação do modelo
            validate_model(fisher_vectors, targets, d_ctrl, f_ctrl, c_ctrl, k, N)

            # Atualizar a barra de progresso após a classificação de cada combinação
            pbar.update(1)

def read_processed_combinations(csv_file):
    processed_combinations = set()
    if os.path.exists(csv_file):
        try:
            with open(csv_file, mode='r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    d_ctrl = int(row['d'])
                    f_ctrl = int(row['f'])
                    c_ctrl = int(row['c'])
                    k = int(row['n_modes'])
                    N = int(row['thre_inc'])
                    processed_combinations.add((d_ctrl, f_ctrl, c_ctrl, k, N))
        except Exception as e:
            print(f"Erro ao ler o arquivo CSV: {e}")
    return processed_combinations

def validate_model(fisher_vectors, targets, d_ctrl, f_ctrl, c_ctrl, k, N):

    # Verifique o número de amostras na menor classe
    min_class_count = min([targets.count(t) for t in set(targets)])
    n_splits = min(10, min_class_count)  # Ajuste n_splits para evitar o erro

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    svm_accuracies = []
    lda_accuracies = []

    for fold, (train_index, test_index) in enumerate(skf.split(fisher_vectors, targets), 1):
        train_fvs = [fisher_vectors[i] for i in train_index]
        test_fvs = [fisher_vectors[i] for i in test_index]
        train_targets = [targets[i] for i in train_index]
        test_targets = [targets[i] for i in test_index]

        # Treinando SVM
        svm = LinearSVC(random_state=42, max_iter=10000, dual="auto")
        svm.fit(train_fvs, train_targets)
        predictions_svm = svm.predict(test_fvs)
        svm_accuracy = accuracy_score(test_targets, predictions_svm)
        svm_accuracies.append(svm_accuracy)

        # Treinando LDA
        lda = LinearDiscriminantAnalysis(solver="eigen", store_covariance=True, shrinkage="auto")
        lda.fit(train_fvs, train_targets)
        predictions_lda = lda.predict(test_fvs)
        lda_accuracy = accuracy_score(test_targets, predictions_lda)
        lda_accuracies.append(lda_accuracy)

    svm_mean_accuracy = np.mean(svm_accuracies)
    svm_std_accuracy = np.std(svm_accuracies)
    lda_mean_accuracy = np.mean(lda_accuracies)
    lda_std_accuracy = np.std(lda_accuracies)

    save_results(d_ctrl, f_ctrl, c_ctrl, N, k, len(fisher_vectors[0]), svm_mean_accuracy, svm_std_accuracy, lda_mean_accuracy, lda_std_accuracy)

def save_results(d_ctrl, f_ctrl, c_ctrl, N, k, num_fisher_vectors, svm_mean_accuracy, svm_std_accuracy, lda_mean_accuracy, lda_std_accuracy):
    with open(results_csv, mode="a", newline="") as csvfile:
        fieldnames = ["d", "f", "c", "thre_inc", "n_modes", "n_fvs", "svm_acc", "svm_std", "lda_acc", "lda_std"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Verificar se o arquivo está vazio para escrever o cabeçalho
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({
            "d": d_ctrl,
            "f": f_ctrl,
            "c": c_ctrl,
            "thre_inc": N,
            "n_modes": k,
            "n_fvs": num_fisher_vectors,
            "svm_acc": f"{svm_mean_accuracy:.4f}",
            "svm_std": f"{svm_std_accuracy:.4f}",
            "lda_acc": f"{lda_mean_accuracy:.4f}",
            "lda_std": f"{lda_std_accuracy:.4f}",
        })

if __name__ == "__main__":
    main()
