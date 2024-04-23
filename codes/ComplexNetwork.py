import numpy as np
from scipy.spatial.distance import pdist, squareform


class ComplexNetwork:
    def __init__(self, thresholding):
        """
        Inicializa a classe CNDescriptorsClass com uma imagem e valores de thresholding.

        :param img: Imagem binária da qual características são extraídas (contorno branco, fundo preto).
        :param thresholding: Valores de thresholding para a extração de características.
        """
        self.thresholding = thresholding

    def extract_features(self, img):
        """
        Extrai características da imagem.

        :return: Tupla de grade, força e coeficiente de clustering.
        """
        coord = np.argwhere(
            img > 0
        )  # Encontra coordenadas dos pixels não-zero (brancos) na imagem.
        cn = self._cn_make(coord)  # Modelagem de rede.
        grade, force, cc = self._thresholding_cn(
            cn, self.thresholding
        )  # CN evoluindo com a seleção de thresholding.

        # Normalização de grade e força.
        grade = grade / grade.shape[1]
        force = force / force.shape[1]
        cc = cc / cc.shape[1]

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
        :return: Tupla de grade, força e coeficiente de clustering.
        """
        cnU = np.zeros_like(cn)
        cnW = np.zeros_like(cn)
        grade = []
        force = []
        cc = []

        for x in thre:
            c = cn < x
            cnU[c] = 1
            cnW[c] = cn[c]
            grade.append(np.sum(cnU, axis=0) - 1)
            force.append(np.sum(cnW, axis=0))
            cc.append(self._coeficiente_clustering(cnU))

        return np.array(grade), np.array(force), np.array(cc)

    ## REVISAR E VERIFICAR SE O CÁCULO ESTÁ CORRETO!!!!
    def _coeficiente_clustering(self, A):
        """
        Calcula o coeficiente de clustering para cada nó da rede.

        :param A: Matriz de adjacência da rede.
        :return: Coeficiente de clustering.
        """
        n = A.shape[0]  # Número de nós.
        A_quadrado = np.dot(A, A)
        A_cubo = np.dot(A_quadrado, A)
        diagonais_cubo = np.diag(A_cubo) / 2

        graus = A.sum(axis=1)  # Grau de cada nó.
        possiveis_conexoes = graus * (graus - 1) / 2
        C = np.zeros(n)  # Inicializa o coeficiente de clustering para cada nó.

        with np.errstate(divide="ignore", invalid="ignore"):
            C = diagonais_cubo / possiveis_conexoes
            C[np.isnan(C)] = 0  # Substitui NaN por 0.

        return C
