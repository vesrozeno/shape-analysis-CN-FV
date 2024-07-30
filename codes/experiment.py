#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para extrair características de imagens usando Redes Complexas e Vetores de Fisher,
seguido de classificação com SVM e LDA.

Criado em: 21 Março 2024
Autores: Lucas C. Ribas e Vitor Emanuel S. Rozeno
"""
# pip install -r requirements.txt

# Importações necessárias
import numpy as np
import glob
import cv2
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from ComplexNetwork import ComplexNetwork
from FisherVectorEncoding import FisherVectorEncoding


# Função principal
def main():
    # Caminho do diretório contendo as imagens
    image_directory = "datasets/eth80/"
    pattern = image_directory + "*.png"

    # Encontrando todos os caminhos de imagem que correspondem ao padrão
    img_paths = glob.glob(pattern)
    print(f"Número total de imagens encontradas: {len(img_paths)}")

    # Definindo as possíveis combinações de parâmetros
    d_ctrl_values = [0, 1]
    f_ctrl_values = [0, 1]
    c_ctrl_values = [0, 1]
    # k_values = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    k_values = [4, 8, 12, 16, 20]
    # N_values = [10, 20, 25, 30, 35, 40, 45, 50]
    N_values = [15, 20, 30, 40, 50]
    
    # Lendo combinações já processadas
    processed_combinations = read_processed_combinations("codes/results_eth.csv")
    print(f"Número de combinações já processadas: {len(processed_combinations)}")

    # Loop sobre todas as combinações de parâmetros com animação de loading
    for d_ctrl in d_ctrl_values:
        for f_ctrl in f_ctrl_values:
            for c_ctrl in c_ctrl_values:
                if d_ctrl == 0 and f_ctrl == 0 and c_ctrl == 0:
                    continue  # Pelo menos um dos controles deve ser 1
                for k in k_values:
                    for N in N_values:
                        # Verificar se a combinação já foi processada
                        if (d_ctrl, f_ctrl, c_ctrl, k, N) in processed_combinations:
                            continue

                        # Valores de limiar para extração de características
                        inc = 1 / N
                        thresholding = np.arange(inc, 1, inc)

                        # Mostrar animação de loading
                        print(f"Processando combinação d_ctrl={d_ctrl}, f_ctrl={f_ctrl}, c_ctrl={c_ctrl}, k={k}, N={N}...")
                        
                        # Processar com a combinação atual de parâmetros
                        process_combination(img_paths, d_ctrl, f_ctrl, c_ctrl, k, N, thresholding)


def read_processed_combinations(csv_file):
    processed_combinations = set()
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
    except FileNotFoundError:
        # Se o arquivo não existe, retornamos um conjunto vazio
        pass
    return processed_combinations


def process_combination(img_paths, d_ctrl, f_ctrl, c_ctrl, k, N, thresholding):
    # Inicializando listas para armazenar os descritores e rótulos das imagens
    degrees, forces, clustering, targets = [], [], [], []

    # Instanciando a classe ComplexNetwork
    CN = ComplexNetwork(thresholding)

    # Usando tqdm para mostrar a barra de progresso
    with tqdm(total=len(img_paths), desc="Processando imagens") as pbar:
        for img_path in img_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            grade, force, cc = CN.extract_features(d_ctrl, f_ctrl, c_ctrl, img)

            # Armazenando os descritores extraídos
            degrees.append(grade.T)
            forces.append(force.T)
            clustering.append(cc.T)

            # Extraindo o rótulo da imagem a partir do nome do arquivo
            targets.append(img_path.split("/")[-1].split("_")[0])

            # Atualizando a barra de progresso
            pbar.update(1)

    # Exibindo o número total de rótulos processados
    tqdm.write(f"Número total de rótulos processados: {len(targets)}")

    # Determinando as medidas utilizadas como features na RC
    CNFeatures = select_features(d_ctrl, f_ctrl, c_ctrl, degrees, forces, clustering)
    tqdm.write(f"Características selecionadas para d_ctrl={d_ctrl}, f_ctrl={f_ctrl}, c_ctrl={c_ctrl}")

    # Instanciando a classe FisherVectorEncoding
    fisher_encoding = FisherVectorEncoding(k)

    # Configuração da validação cruzada estratificada
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Listas para armazenar resultados globais
    overall_targets, svm_overall_predictions, svm_accuracies = [], [], []
    lda_overall_predictions, lda_accuracies = [], []

    # Loop de validação cruzada
    for train_index, test_index in skf.split(CNFeatures, targets):
        train_descriptors = [CNFeatures[i] for i in train_index]
        test_descriptors = [CNFeatures[i] for i in test_index]
        train_targets = [targets[i] for i in train_index]
        test_targets = [targets[i] for i in test_index]

        # Computando os vetores de Fisher para os conjuntos de treino e teste
        training_fvs, testing_fvs = fisher_encoding.compute_fisher_vectors(
            train_descriptors, test_descriptors
        )

        # Treinando e avaliando o modelo SVM
        svm = LinearSVC(random_state=42, max_iter=10000, dual="auto")
        svm.fit(training_fvs, train_targets)
        predictions_svm = svm.predict(testing_fvs)

        # Treinando e avaliando o modelo LDA
        lda = LinearDiscriminantAnalysis(
            solver="eigen", store_covariance=True, shrinkage="auto"
        )
        lda.fit(training_fvs, train_targets)
        predictions_lda = lda.predict(testing_fvs)

        # Armazenando resultados para análise posterior
        overall_targets.extend(test_targets)

        # SVM
        svm_overall_predictions.extend(predictions_svm)
        svm_accuracies.append(accuracy_score(test_targets, predictions_svm))

        # LDA
        lda_overall_predictions.extend(predictions_lda)
        lda_accuracies.append(accuracy_score(test_targets, predictions_lda))

    # Calculando e exibindo a acurácia média e o desvio padrão - SVM
    svm_mean_accuracy = np.mean(svm_accuracies)
    svm_std_accuracy = np.std(svm_accuracies)
    tqdm.write(
        f"SVM Mean accuracy over 10 folds for d_ctrl={d_ctrl}, f_ctrl={f_ctrl}, c_ctrl={c_ctrl}, k={k}, N={N}: {svm_mean_accuracy:.4f} ± {svm_std_accuracy:.4f}"
    )

    # Calculando e exibindo a acurácia média e o desvio padrão - LDA
    lda_mean_accuracy = np.mean(lda_accuracies)
    lda_std_accuracy = np.std(lda_accuracies)
    tqdm.write(
        f"LDA Mean accuracy over 10 folds for d_ctrl={d_ctrl}, f_ctrl={f_ctrl}, c_ctrl={c_ctrl}, k={k}, N={N}: {lda_mean_accuracy:.4f} ± {lda_std_accuracy:.4f}"
    )

    # Salvando os resultados em um arquivo CSV
    save_results(
        d_ctrl,
        f_ctrl,
        c_ctrl,
        N,
        k,
        svm_mean_accuracy,
        svm_std_accuracy,
        lda_mean_accuracy,
        lda_std_accuracy,
    )
    tqdm.write(f"Resultados salvos para d_ctrl={d_ctrl}, f_ctrl={f_ctrl}, c_ctrl={c_ctrl}, k={k}, N={N}")



def select_features(d_ctrl, f_ctrl, c_ctrl, degrees, forces, clustering):
    """
    Seleciona as features a serem utilizadas na RC com base nos parâmetros de controle.

    :param d_ctrl: Controle do grau
    :param f_ctrl: Controle da força
    :param c_ctrl: Controle do coeficiente de clustering
    :param degrees: Lista de graus
    :param forces: Lista de forças
    :param clustering: Lista de coeficientes de clustering
    :return: Lista de características combinadas
    """
    if d_ctrl == 1 and f_ctrl == 0 and c_ctrl == 0:
        return degrees
    elif d_ctrl == 0 and f_ctrl == 1 and c_ctrl == 0:
        return forces
    elif d_ctrl == 0 and f_ctrl == 0 and c_ctrl == 1:
        return clustering
    elif d_ctrl == 1 and f_ctrl == 1 and c_ctrl == 0:
        return [np.concatenate([d, f], axis=1) for d, f in zip(degrees, forces)]
    elif d_ctrl == 1 and f_ctrl == 0 and c_ctrl == 1:
        return [np.concatenate([d, c], axis=1) for d, c in zip(degrees, clustering)]
    elif d_ctrl == 0 and f_ctrl == 1 and c_ctrl == 1:
        return [np.concatenate([f, c], axis=1) for f, c in zip(forces, clustering)]
    elif d_ctrl == 1 and f_ctrl == 1 and c_ctrl == 1:
        return [
            np.concatenate([d, f, c], axis=1)
            for d, f, c in zip(degrees, forces, clustering)
        ]


def save_results(
    d_ctrl,
    f_ctrl,
    c_ctrl,
    N,
    k,
    svm_mean_accuracy,
    svm_std_accuracy,
    lda_mean_accuracy,
    lda_std_accuracy,
):
    """
    Salva os resultados em um arquivo CSV.

    :param d_ctrl: Controle do grau
    :param f_ctrl: Controle da força
    :param c_ctrl: Controle do coeficiente de clustering
    :param N: Número de limiares
    :param k: Número de componentes do GMM
    :param svm_mean_accuracy: Acurácia média do SVM
    :param svm_std_accuracy: Desvio padrão da acurácia do SVM
    :param lda_mean_accuracy: Acurácia média do LDA
    :param lda_std_accuracy: Desvio padrão da acurácia do LDA
    """
    with open("codes/results_eth.csv", mode="a", newline="") as csvfile:
        fieldnames = [
            "d",
            "f",
            "c",
            "thre_inc",
            "n_modes",
            "svm_acc",
            "svm_std",
            "lda_acc",
            "lda_std",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow(
            {
                "d": d_ctrl,
                "f": f_ctrl,
                "c": c_ctrl,
                "thre_inc": N,
                "n_modes": k,
                "svm_acc": f"{svm_mean_accuracy:.4f}",
                "svm_std": f"{svm_std_accuracy:.4f}",
                "lda_acc": f"{lda_mean_accuracy:.4f}",
                "lda_std": f"{lda_std_accuracy:.4f}",
            }
        )


if __name__ == "__main__":
    main()
