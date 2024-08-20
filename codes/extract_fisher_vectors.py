#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para extrair características de imagens usando Redes Complexas e Vetores de Fisher.

Criado em: 21 Março 2024
Autores: Lucas C. Ribas e Vitor Emanuel S. Rozeno
"""

import os
import numpy as np
import glob
import cv2
import pickle
import shutil
from joblib import Parallel, delayed

from ComplexNetwork import ComplexNetwork
from FisherVectorEncoding import FisherVectorEncoding

# Variáveis para determinar o nome do arquivo pickle e do dataset
image_directory = "datasets/LeavesPortuguese/"  # Caminho do diretório contendo as imagens
fisher_vectors_pkl = "pkl/fisher_vectors_and_targets_portuguese.pkl"  # Nome do arquivo pkl para salvar os Fisher Vectors e rótulos

def main(num_cores):
    pattern = image_directory + "*.bmp"

    # Encontrando todos os caminhos de imagem que correspondem ao padrão
    img_paths = glob.glob(pattern)
    print(f"Número total de imagens encontradas: {len(img_paths)}")

    # Extração das características e cálculo dos Fisher Vectors
    extract_and_save_fisher_vectors(img_paths, num_cores)

def extract_and_save_fisher_vectors(img_paths, num_cores):
    # Carregar dados existentes do arquivo .pkl, se disponível
    if os.path.exists(fisher_vectors_pkl):
        with open(fisher_vectors_pkl, "rb") as f:
            existing_data = pickle.load(f)
            fisher_vectors_dict = existing_data.get("fisher_vectors", {})
            targets = existing_data.get("targets", [])
            feature_data = existing_data.get("feature_data", {})  # Dados de características existentes
    else:
        fisher_vectors_dict = {}
        targets = []
        feature_data = {}

    # Definindo as possíveis combinações de parâmetros
    d_ctrl_values = [0, 1]
    f_ctrl_values = [0, 1]
    c_ctrl_values = [0, 1]
    k_values = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    N_values = [10, 20, 25, 30, 35, 40, 45, 50]

    # Verificando se já existem targets no arquivo
    if len(targets) < len(img_paths):
        # Extraindo os rótulos da imagem a partir do nome do arquivo
        for img_path in img_paths[len(targets):]:  # Somente adiciona novos targets se necessário
            targets.append(img_path.split("/")[-1].split("_")[0])
        print(f"Targets atualizados: {len(targets)}")
    else:
        print(f"Targets já completos: {len(targets)}")

    # Loop sobre todas as combinações de parâmetros
    for N in N_values:
        # Valores de limiar para extração de características
        inc = 1 / N
        thresholding = np.arange(inc, 1, inc)

        # Instanciando a classe ComplexNetwork
        CN = ComplexNetwork(thresholding)

        # Verificar se as características para esse threshold já foram calculadas
        if N in feature_data:
            degrees, forces, clustering = feature_data[N]['degrees'], feature_data[N]['forces'], feature_data[N]['clustering']
        else:
            # Inicializando listas para os descritores dessa combinação
            degrees, forces, clustering = [], [], []

            print(f"Iniciando extração de features para N={N}...")
            results = Parallel(n_jobs=num_cores)(
                delayed(CN.extract_features)(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)) for img_path in img_paths
            )

            for d, f, c in results:
                degrees.append(d.T)
                forces.append(f.T)
                clustering.append(c.T)

            print(f"Features extraídas para N={N}.")

            # Adicionando as características extraídas para essa configuração de N ao dicionário
            feature_data[N] = {
                'degrees': degrees,
                'forces': forces,
                'clustering': clustering
            }

            # Salvando o arquivo pickle após a extração de features para o threshold N
            save_data(fisher_vectors_dict, targets, feature_data, "feature_data")

        print(f"Iniciando cálculo dos Fisher Vectors para N={N}...")
        for d_ctrl in d_ctrl_values:
            for f_ctrl in f_ctrl_values:
                for c_ctrl in c_ctrl_values:
                    if d_ctrl == 0 and f_ctrl == 0 and c_ctrl == 0:
                        continue  # Pelo menos um dos controles deve ser 1
                    for k in k_values:
                        # Verificar se a combinação já foi processada
                        if (d_ctrl, f_ctrl, c_ctrl, k, N) in fisher_vectors_dict:
                            continue

                        # Combinar as características de acordo com os parâmetros de controle
                        CNFeatures = combine_features(d_ctrl, f_ctrl, c_ctrl, degrees, forces, clustering)

                        # Instanciando a classe FisherVectorEncoding
                        fisher_encoding = FisherVectorEncoding(k)

                        # Computando os Fisher Vectors
                        fisher_vectors, _ = fisher_encoding.compute_fisher_vectors(CNFeatures, CNFeatures)

                        # Salvando os Fisher Vectors dessa combinação no dicionário
                        fisher_vectors_dict[(d_ctrl, f_ctrl, c_ctrl, k, N)] = fisher_vectors

        # Salvando o arquivo pickle após o cálculo do Fisher Vector
        save_data(fisher_vectors_dict, targets, feature_data, "fisher_vectors")

def save_data(fisher_vectors_dict, targets, feature_data, changed_data):
    """
    Função para salvar os dados no arquivo pickle se houver novas informações para salvar.
    Cria um backup do arquivo existente antes de sobrescrevê-lo.
    """
    # Se o arquivo pickle já existir, cria um backup
    if os.path.exists(fisher_vectors_pkl):
        backup_path = fisher_vectors_pkl + ".bak"
        shutil.copy2(fisher_vectors_pkl, backup_path)
        print(f"Backup do arquivo criado em '{backup_path}'.")

    # Se o arquivo pickle não existir, cria um arquivo vazio
    if not os.path.exists(fisher_vectors_pkl):
        with open(fisher_vectors_pkl, "wb") as f:
            pickle.dump({
                "fisher_vectors": {},
                "targets": [],
                "feature_data": {}
            }, f)

    # Carregar dados anteriores do arquivo pickle
    with open(fisher_vectors_pkl, "rb") as f:
        existing_data = pickle.load(f)
    
    # Verifica se há novos dados para salvar
    if changed_data == "fisher_vectors":
        if not dicts_are_equal(existing_data.get("fisher_vectors", {}), fisher_vectors_dict):
            with open(fisher_vectors_pkl, "wb") as f:
                pickle.dump({
                    "fisher_vectors": fisher_vectors_dict,
                    "targets": targets,
                    "feature_data": feature_data
                }, f)
            print(f"Novos Fisher Vectors salvos em '{fisher_vectors_pkl}'.")
    elif changed_data == "feature_data":
        if not dicts_are_equal(existing_data.get("feature_data", {}), feature_data):
            with open(fisher_vectors_pkl, "wb") as f:
                pickle.dump({
                    "fisher_vectors": fisher_vectors_dict,
                    "targets": targets,
                    "feature_data": feature_data
                }, f)
            print(f"Novos dados de features salvos em '{fisher_vectors_pkl}'.")

def dicts_are_equal(dict1, dict2):
    """
    Compara dois dicionários que podem conter arrays do NumPy.
    Retorna True se os dicionários forem iguais, caso contrário, False.
    """
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if isinstance(dict1[key], np.ndarray):
            if not np.array_equal(dict1[key], dict2[key]):
                return False
        elif isinstance(dict1[key], dict):
            if not dicts_are_equal(dict1[key], dict2[key]):
                return False
        else:
            if dict1[key] != dict2[key]:
                return False
    return True

def combine_features(d_ctrl, f_ctrl, c_ctrl, degrees, forces, clustering):
    """
    Combina características conforme os parâmetros de controle.

    :param d_ctrl: Controle do grau
    :param f_ctrl: Controle da força
    :param c_ctrl: Controle do coeficiente de clustering
    :param degrees: Lista de graus para todos os thresholds
    :param forces: Lista de forças para todos os thresholds
    :param clustering: Lista de coeficientes de clustering para todos os thresholds
    :return: Lista de características combinadas
    """
    combined_features = []
    for d, f, c in zip(degrees, forces, clustering):
        feature_set = []
        if d_ctrl:
            feature_set.append(d)
        if f_ctrl:
            feature_set.append(f)
        if c_ctrl:
            feature_set.append(c)
        combined_features.append(np.concatenate(feature_set, axis=0))
    return combined_features

if __name__ == "__main__":
    num_cores = int(input("Digite o número de núcleos para processamento paralelo: "))
    main(num_cores=num_cores)
