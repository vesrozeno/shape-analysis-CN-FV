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
from tqdm import tqdm

from ComplexNetwork import ComplexNetwork
from FisherVectorEncoding import FisherVectorEncoding

# Variáveis para determinar o nome do arquivo pickle e do dataset
image_directory = "datasets/eth80/"  # Caminho do diretório contendo as imagens
fisher_vectors_pkl = "pkl/eth6020.pkl"  # Nome do arquivo pkl para salvar os Fisher Vectors e rótulos

def main():
    pattern = image_directory + "*.png"

    # Encontrando todos os caminhos de imagem que correspondem ao padrão
    img_paths = glob.glob(pattern)
    print(f"Número total de imagens encontradas: {len(img_paths)}")

    # Extração das características e cálculo dos Fisher Vectors
    extract_and_save_fisher_vectors(img_paths)

def extract_and_save_fisher_vectors(img_paths):
    # Carregar dados existentes do arquivo .pkl, se disponível
    if os.path.exists(fisher_vectors_pkl):
        with open(fisher_vectors_pkl, "rb") as f:
            existing_data = pickle.load(f)
            fisher_vectors_dict = existing_data.get("fisher_vectors", {})
            targets = existing_data.get("targets", [])
            targets = []
            feature_data = existing_data.get("feature_data", {})  # Dados de características existentes
    else:
        fisher_vectors_dict = {}
        targets = []
        feature_data = {}

    # Definindo as possíveis combinações de parâmetros
    d_ctrl_values = [0, 1]
    f_ctrl_values = [0, 1]
    c_ctrl_values = [0, 1]
    k_values = [4, 8, 10, 14, 18, 20, 24]
    N_values = [35, 45, 60, 70]

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
            # print(f"Características já calculadas para N={N}. Pulando...")
            degrees, forces, clustering = feature_data[N]['degrees'], feature_data[N]['forces'], feature_data[N]['clustering']
        else:
            # Inicializando listas para os descritores dessa combinação
            degrees, forces, clustering = [], [], []

            # Usando tqdm para mostrar a barra de progresso
            with tqdm(total=len(img_paths), desc=f"Extraindo features para N={N}") as pbar:
                for img_path in img_paths:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Calcular características para todos os thresholds
                    d, f, c = CN.extract_features(img)

                    degrees.append(d.T)
                    forces.append(f.T)
                    clustering.append(c.T)
                    # Atualizando a barra de progresso
                    pbar.update(1)
            
            # Adicionando as características extraídas para essa configuração de N ao dicionário
            feature_data[N] = {
                'degrees': degrees,
                'forces': forces,
                'clustering': clustering
            }

            # Salvando o arquivo pickle após a extração de features para o threshold N
            save_data(fisher_vectors_dict, targets, feature_data)

        with tqdm(total=len(d_ctrl_values)*len(f_ctrl_values)*len(c_ctrl_values)*len(k_values)-len(k_values), 
                  desc=f"Calculando Fisher Vectors para N={N}") as wbar:
            for d_ctrl in d_ctrl_values:
                for f_ctrl in f_ctrl_values:
                    for c_ctrl in c_ctrl_values:
                        if d_ctrl == 0 and f_ctrl == 0 and c_ctrl == 0:
                            continue  # Pelo menos um dos controles deve ser 1
                        for k in k_values:
                             # Verificar se a combinação já foi processada
                            if (d_ctrl, f_ctrl, c_ctrl, k, N) in fisher_vectors_dict:
                                # print(f"Combinação já processada: d_ctrl={d_ctrl}, f_ctrl={f_ctrl}, c_ctrl={c_ctrl}, k={k}, N={N}. Pulando...")
                                wbar.update(1)
                                continue

                            # Combinar as características de acordo com os parâmetros de controle
                            CNFeatures = combine_features(d_ctrl, f_ctrl, c_ctrl, degrees, forces, clustering)

                            # Instanciando a classe FisherVectorEncoding
                            fisher_encoding = FisherVectorEncoding(k)

                            # Computando os Fisher Vectors
                            fisher_vectors, _ = fisher_encoding.compute_fisher_vectors(CNFeatures, CNFeatures)

                            # Salvando os Fisher Vectors dessa combinação no dicionário
                            fisher_vectors_dict[(d_ctrl, f_ctrl, c_ctrl, k, N)] = fisher_vectors

                            # Atualizando a barra de progresso
                            wbar.update(1)
            # Salvando o arquivo pickle após o cálculo do Fisher Vector
            save_data(fisher_vectors_dict, targets, feature_data)

def save_data(fisher_vectors_dict, targets, feature_data):
    """
    Função para salvar os dados no arquivo pickle.
    """
    with open(fisher_vectors_pkl, "wb") as f:
        pickle.dump({
            "fisher_vectors": fisher_vectors_dict,
            "targets": targets,
            "feature_data": feature_data
        }, f)
    print(f"\nDados salvos em '{fisher_vectors_pkl}'.")

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
        features = []
        if d_ctrl:
            features.append(d)
        if f_ctrl:
            features.append(f)
        if c_ctrl:
            features.append(c)
        combined_features.append(np.concatenate(features, axis=1))
    return combined_features

if __name__ == "__main__":
    main()