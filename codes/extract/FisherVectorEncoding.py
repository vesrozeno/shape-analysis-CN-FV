#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 23:26:12 2024

@author: lucas
"""

import numpy as np
from skimage.feature import fisher_vector, learn_gmm


class FisherVectorEncoding:

    def __init__(self, n_modes):
        """
        Inicializa a classe ImageClassifier.

        :param n_modes: Número de modos para a Gaussian Mixture Model (GMM).
        """
        self.n_modes = n_modes
        self.gmm = None

    def learn_gmm(self, descriptors):
        """
        Treina a Gaussian Mixture Model (GMM) nos descritores fornecidos.

        :param descriptors: Descritores de treino.
        """
        self.gmm = learn_gmm(descriptors, n_modes=self.n_modes)

    def compute_fisher_vectors_unique(self, descriptors):
        """
        Calcula os vetores de Fisher para os descritores fornecidos.

        :param descriptors: Descritores para os quais calcular os vetores de Fisher.
        :return: Vetores de Fisher calculados.
        """
        if self.gmm is None:
            raise ValueError("GMM has not been trained. Call 'learn_gmm' first.")
        return np.array(
            [fisher_vector(descriptor, self.gmm) for descriptor in descriptors]
        )

    def compute_fisher_vectors(self, train_descriptors, test_descriptors):
        """
        Treina a GMM, calcula vetores de Fisher para treino e teste, e realiza a classificação.

        :param train_descriptors: Descritores de treino.
        :param train_labels: Rótulos de treino.
        :param test_descriptors: Descritores de teste.
        :param test_labels: Rótulos de teste.
        """
        # Treinando a GMM nos descritores de treino
        self.learn_gmm(train_descriptors)

        # Calculando vetores de Fisher para treino e teste
        training_fvs = self.compute_fisher_vectors_unique(train_descriptors)
        testing_fvs = self.compute_fisher_vectors_unique(test_descriptors)

        return training_fvs, testing_fvs
