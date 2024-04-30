#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para extrair características de imagens usando Redes Complexas e Vetores de Fisher,
seguido de classificação com SVM.

Criado em: 21 Março 2024
Autor: lucas
"""

# Importações necessárias
import numpy as np
import glob
import cv2
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ComplexNetwork import ComplexNetwork
from FisherVectorEncoding import FisherVectorEncoding

# Caminho do diretório contendo as imagens
image_directory = "datasets/Leaves256x256c/"
# image_directory = "datasets/teste/"
pattern = image_directory + "*.png"

# Encontrando todos os caminhos de imagem que correspondem ao padrão
img_paths = glob.glob(pattern)

# Inicializando listas para armazenar os descritores e rótulos das imagens
degrees = []
forces = []
clustering = []
targets = []

# Definindo medidas da RC
d_ctrl = 0
f_ctrl = 0
c_ctrl = 1

# Número de componentes para o modelo GMM
k = 4

# Valores de limiar para extração de características
# thresholding = np.linspace(0.025, 0.95, num=10)

# Definindo o valor de N
N = 30

# Calculando o incremento
inc = 1 / N

# Arange NumPy, para gerar os valores excluindo 0 e 1
thresholding = np.arange(inc, 1, inc)


# Instanciando a classe ComplexNetwork
CN = ComplexNetwork(thresholding)

# Extração de características de redes compplexas para todas as imagens
for img_path in img_paths:
    # PARA IMAGEM BINÁRIA (FUNDO PRETO (O) CONTORNO BRANCO (255))
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    grade, force, cc = CN.extract_features(img)

    # Armazenando os descritores extraídos
    degrees.append(grade.T)
    forces.append(force.T)
    clustering.append(cc.T)

    # Extraindo o rótulo da imagem a partir do nome do arquivo. Pode mudar dependendo do dataset
    targets.append(img_path.split("/")[-1].split("_")[0])

#Determinando as medidas utlizadas como features na RC
# Combinando graus e forças em um único conjunto de características

if d_ctrl == 1 and f_ctrl == 0 and c_ctrl == 0:
    CNFeatures = forces
elif d_ctrl == 0 and f_ctrl == 1 and c_ctrl == 0:
    CNFeatures = degrees
elif d_ctrl == 0 and f_ctrl == 0 and c_ctrl == 1:
    CNFeatures = clustering
elif d_ctrl == 1 and f_ctrl == 1 and c_ctrl == 0:
    CNFeatures = [np.concatenate([d, f], axis=1) for d, f in zip(degrees, forces)]
elif d_ctrl == 1 and f_ctrl == 0 and c_ctrl == 1:
    CNFeatures = [np.concatenate([d, c], axis=1) for d, c in zip(degrees, clustering)]
elif d_ctrl == 0 and f_ctrl == 1 and c_ctrl == 1:
    CNFeatures = [np.concatenate([f, c], axis=1) for f, c in zip(forces, clustering)]
elif d_ctrl == 1 and f_ctrl == 1 and c_ctrl == 1:
    CNFeatures = [
        np.concatenate([d, f, c], axis=1)
        for d, f, c in zip(degrees, forces, clustering)
    ]

# Instanciando a classe FisherVectorEncoding
fisher_encoding = FisherVectorEncoding(k)

# Configuração da validação cruzada estratificada
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Listas para armazenar resultados globais
overall_targets = []
svm_overall_predictions = []
svm_accuracies = []
lda_overall_predictions = []
lda_accuracies = []

# Loop de validação cruzada
for train_index, test_index in skf.split(CNFeatures, targets):
    # Separando os dados em conjuntos de treino e teste
    train_descriptors = [CNFeatures[i] for i in train_index]
    test_descriptors = [CNFeatures[i] for i in test_index]
    train_targets = [targets[i] for i in train_index]
    test_targets = [targets[i] for i in test_index]

    # Computando os vetores de Fisher para os conjuntos de treino e teste
    training_fvs, testing_fvs = fisher_encoding.compute_fisher_vectors(
        train_descriptors, test_descriptors
    )

    # Treinando o modelo SVM com os vetores de Fisher de treino
    svm = LinearSVC(
        random_state=42, max_iter=10000, dual="auto"
    )  # Aumentado max_iter para garantir a convergência
    svm.fit(training_fvs, train_targets)

    # Treinando o modelo LDA com os vetores de Fisher de treino
    lda = LinearDiscriminantAnalysis(
        solver="eigen", store_covariance=True, shrinkage="auto"
    )
    lda.fit(training_fvs, train_targets)

    # Realizando predições com o modelo treinado
    predictions_svm = svm.predict(testing_fvs)

    # Predições com LDA
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
print(
    f"SVM Mean accuracy over 10 folds: {svm_mean_accuracy:.4f} ± {svm_std_accuracy:.4f}"
)

# Calculando e exibindo a acurácia média e o desvio padrão - LDA
lda_mean_accuracy = np.mean(lda_accuracies)
lda_std_accuracy = np.std(lda_accuracies)
print(
    f"LDA Mean accuracy over 10 folds: {lda_mean_accuracy:.4f} ± {lda_std_accuracy:.4f}"
)


# Gerando e exibindo o relatório de classificação
# print(classification_report(overall_targets, svm_overall_predictions))


#  Salvar os resultados em um arquivo CSV
with open("codes/results.csv", mode="a") as csvfile:
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
   
    # increment_description = np.diff(thresholding.tolist())[0]
    writer.writerow(
        {
            "d": d_ctrl,
            "f": f_ctrl,
            "c": c_ctrl,
            "thre_inc": f"0:{inc:.4f}:1",
            "n_modes": k,
            "svm_acc": f"{svm_mean_accuracy:.4f}",
            "svm_std": f"{svm_std_accuracy:.4f}",
            "lda_acc": f"{lda_mean_accuracy:.4f}",
            "lda_std": f"{lda_std_accuracy:.4f}",
        }
    )
