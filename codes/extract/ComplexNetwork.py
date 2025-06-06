import numpy as np
from scipy.spatial.distance import pdist, squareform


class ComplexNetwork:
    def __init__(self, thresholding):
        """
        Inicializa a classe ComplexNetwork com valores de thresholding.

        :param thresholding: Valores de thresholding para a extração de características.
        """
        self.thresholding = thresholding

    def extract_features(self, img):
        """
        Extrai características da imagem.

        :param img: Imagem binária da qual características são extraídas (contorno branco, fundo preto).
        :return: Tupla de grau, força e coeficiente de clustering.
        """
        coord = np.argwhere(
            img > 0
        )  # Encontra coordenadas dos pixels não-zero (brancos) na imagem.
        cn = self._cn_make(coord)  # Modelagem de rede.
        grade, force, cc = self._thresholding_cn(
             cn, self.thresholding
        )  # CN evoluindo com a seleção de thresholding.

        # Normalização de grau e força.
        grade = grade / grade.shape[1]
        force = force / force.shape[1]

        return grade, force, cc

    def _cn_make(self, coord):
        """
        Cria a rede complexa a partir das coordenadas.

        :param coord: Coordenadas dos pixels não-zero na imagem.
        :return: Rede complexa CN.
        """
        y = pdist(coord, "euclidean")  # Calcula distâncias Euclidianas par a par.
        CN = squareform(y)
        CN = CN / np.max(CN)  # Normaliza.

        return CN

    def _thresholding_cn(self, cn, thre):
        """
        Aplica thresholding à rede complexa.

        :param cn: Rede complexa.
        :param thre: Valores de thresholding.
        :return: Tupla de grau, força e coeficiente de clustering.
        """
        cnU = np.zeros_like(cn)
        cnW = np.zeros_like(cn)
        grade = []
        force = []
        cc = []

        for x in thre:
            c = cn < x
            cnU[c] = 1
            grade.append(np.sum(cnU, axis=0) - 1)
            cnW[c] = cn[c]
            force.append(np.sum(cnW, axis=0))
            cc.append(self._clustering(cnU))

        return np.array(grade), np.array(force), np.array(cc)

    def _clustering(self, A):
            """
            Calcula o coeficiente de clustering para cada nó da rede.

            :param A: Matriz de adjacência da rede.
            :return: Coeficiente de clustering.
            """
            triangles = np.dot(A, A) * A  # Número de triângulos passados por cada aresta.
            triangles = np.sum(triangles, axis=1) / 2  # Divide por 2 para contar corretamente.
            
            degrees = np.sum(A, axis=1)  # Grau de cada nó.
            clustering_coeffs = np.zeros(A.shape[0])
            
            # Coeficiente de clustering é o número de triângulos dividido pelo número possível de triângulos.
            possible_triangles = degrees * (degrees - 1) / 2
            nonzero_possible = possible_triangles > 0
            clustering_coeffs[nonzero_possible] = triangles[nonzero_possible] / possible_triangles[nonzero_possible]
            
            return clustering_coeffs
    