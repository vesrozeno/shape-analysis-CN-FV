# Importações necessárias
import numpy as np
from ComplexNetwork import ComplexNetwork
from FisherVectorEncoding import FisherVectorEncoding
from sklearn.mixture import GaussianMixture
from skimage.feature import fisher_vector

# Inicializando listas para armazenar os descritores
degrees = []
forces = []
clustering = []

# Número de componentes para o modelo GMM
k = 2

# Definindo o valor de N
N = 5

# Calculando o incremento
inc = 1 / N

# Arange NumPy, para gerar os valores excluindo 0 e 1
thresholding = np.arange(inc, 1, inc)

# Instanciando a classe ComplexNetwork
CN = ComplexNetwork(thresholding)

# Exemplo de duas matrizes de adjacência
adjacency_matrices = [
    np.array([
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]),
    np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0]
    ]),
]

adjacency_matrix = np.array([
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 0]
])

grau,_,_ = CN.extract_features(adjacency_matrix)

grau = grau.T

print(f'Graus: \n{grau}')


# Extração de características da rede complexa para todas as matrizes de adjacência
for adj_matrix in adjacency_matrices:
    # Extraindo características (graus, forças, coeficiente de clustering)
    grade, force, cc = CN.extract_features(adj_matrix)
    
    # Armazenando os descritores extraídos
    degrees.append(grade.T)
    forces.append(force.T)
    clustering.append(cc.T)

CNFeatures = np.concatenate([np.concatenate([d, f, c], axis=1) for d, f, c in zip(degrees, forces, clustering)], axis=0)

print(f'Descritores - Graus: \n{degrees[0]}')
print(f'Descritores - Graus: \n{len(degrees[0])}')
print(f'CNFeatures: \n{CNFeatures[0]}')
print(f'CNFeatures: \n{len(CNFeatures[0])}')

# Configurações para o Fisher Vector Encoding
fisher_encoding = FisherVectorEncoding(k)


gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='diag')
gmm.fit(CNFeatures)

print(f"gmm means:\n{gmm.means_}")
print(f"gmm covariances:\n{gmm.covariances_}")
print(f"gmm weights:\n{gmm.weights_}")

if grau.ndim == 1:
    grau = grau.reshape(-1, 1)

fv = np.array([fisher_vector(grau, gmm)])

# Computando os vetores de Fisher (apenas como exemplo, usando os mesmos descritores para treino e teste)
# training_fvs, testing_fvs = fisher_encoding.compute_fisher_vectors(CNFeatures, CNFeatures)

print(f'Vetor de Fisher (treino): \n{fv}')
print(f'Vetor de Fisher (treino): \n{fv.shape}')

# Verificar a saída da GMM e dos vetores de Fisher
# print(f'Vetor de Fisher (treino): \n{training_fvs[0]}')
# print(f'Vetor de Fisher (treino): \n{training_fvs.shape}')
