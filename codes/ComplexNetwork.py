import numpy as np
from scipy.spatial.distance import pdist, squareform


class ComplexNetwork:
    def __init__(self, thresholding):
        """
        Inicializa a classe ComplexNetwork com valores de thresholding.

        :param thresholding: Valores de thresholding para a extração de características.
        """
        self.thresholding = thresholding

    def extract_features(self, d_ctrl, f_ctrl, c_ctrl, img):
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
            d_ctrl, f_ctrl, c_ctrl, cn, self.thresholding
        )  # CN evoluindo com a seleção de thresholding.

        # Normalização de grau e força.
        if(d_ctrl):
            grade = grade / grade.shape[1]
        if(f_ctrl):
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

    def _thresholding_cn(self, d_ctrl, f_ctrl, c_ctrl, cn, thre):
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
            if d_ctrl:
                grade.append(np.sum(cnU, axis=0) - 1)
            if f_ctrl:
                cnW[c] = cn[c]
                force.append(np.sum(cnW, axis=0))
            if c_ctrl:
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
    # def _clustering(self, A):
    #     """
    #     Calcula o coeficiente de clustering para cada nó da rede.

    #     :param A: Matriz de adjacência da rede.
    #     :return: Coeficiente de clustering.
    #     """
    #     G = nx.Graph(A)
    #     clustering_coeffs = nx.clustering(G)
    #     # Convertendo o dicionário para uma lista ordenada de coeficientes de clustering
    #     C = np.array([clustering_coeffs[i] for i in range(len(clustering_coeffs))])
    #     return C

    # def _clustering(self, A):
    #     """
    #     Calcula o coeficiente de clustering para cada nó da rede

    #     :param A: Matriz de adjacência da rede.
    #     :return: Coeficiente de clustering.
    #     """
    #     n = A.shape[0]
    #     C = np.zeros(n)

    #     for i in range(n):
    #         neighbors = np.where(A[i] == 1)[0]
    #         k_i = len(neighbors)
    #         if k_i < 2:
    #             C[i] = 0.0
    #             continue

    #         links = 0
    #         for u in neighbors:
    #             for v in neighbors:
    #                 if A[u, v] == 1:
    #                     links += 1

    #         links /= 2  # Cada triângulo é contado duas vezes.
    #         C[i] = (2 * links) / (k_i * (k_i - 1))

    #     return C

