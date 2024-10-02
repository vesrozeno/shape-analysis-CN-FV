# Importações necessárias
import numpy as np
from ComplexNetwork import ComplexNetwork
from FisherVectorEncoding import FisherVectorEncoding
from sklearn.mixture import GaussianMixture

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

# Iterando sobre as matrizes de adjacência
for adj_matrix in adjacency_matrices:
    # Extraindo características (graus, forças, coeficiente de clustering)
    grade, force, cc = CN.extract_features(adj_matrix)
    
    # Armazenando os descritores extraídos
    degrees.append(grade.T)  # Transpondo se necessário
    forces.append(force.T)    # Transpondo se necessário
    clustering.append(cc.T)    # Transpondo se necessário

# Concatenando todas as características extraídas
# Usando a função combine_features para combinar conforme os parâmetros desejados
CNFeatures = np.concatenate([
    np.concatenate([d, f, c], axis=1) 
    for d, f, c in zip(degrees, forces, clustering)
], axis=0)

# Verificando a saída dos descritores
print(f'Descritores - Graus da primeira amostra: \n{degrees[0]}')
print(f'Tamanho dos descritores da primeira amostra: {len(degrees[0])}')
print(f'CNFeatures (todas as características concatenadas): \n{CNFeatures}')
print(f'Tamanho de CNFeatures: {CNFeatures.shape}')

# Configurações para o Fisher Vector Encoding
fisher_encoding = FisherVectorEncoding(k)

# Ajuste do modelo GMM usando as características extraídas
gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='diag')
gmm.fit(CNFeatures)

# Verificar a saída do GMM
print(f"GMM - Means:\n{gmm.means_}")
print(f"GMM - Covariances:\n{gmm.covariances_}")
print(f"GMM - Weights:\n{gmm.weights_}")

CNFeatures = CNFeatures.reshape(1, -1)

# Computando os vetores de Fisher (apenas como exemplo, usando os mesmos descritores para treino e teste)
training_fvs,testing_fvs = fisher_encoding.compute_fisher_vectors(CNFeatures,CNFeatures)

# Verificar a saída dos vetores de Fisher
print(f'Vetor de Fisher (treino): \n{training_fvs[0]}')
print(f'Tamanho do Vetor de Fisher (treino): {training_fvs[0].shape}')