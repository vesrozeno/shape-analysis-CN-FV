import numpy as np
import glob
import cv2
import gc  # Para coleta de lixo manual
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ComplexNetwork import ComplexNetwork
from FisherVectorEncoding import FisherVectorEncoding

# Caminho do diretório contendo as imagens
image_directory = 'datasets/Arquivo/Leaves256x256cs/'
pattern = image_directory + '*.png'

# Encontrando todos os caminhos de imagem que correspondem ao padrão
img_paths = glob.glob(pattern)

# Definindo medidas da RC
d_ctrl = 1
f_ctrl = 1
c_ctrl = 1

# Número de componentes para o modelo GMM
k = 20

# Valores de limiar para extração de características
N = 60
inc = 1 / N
thresholding = np.arange(inc, 1, inc)

# Instanciando a classe ComplexNetwork
CN = ComplexNetwork(thresholding)

# Função para processar uma única imagem
def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    grade, force, cc = CN.extract_features(img)
    img_name = img_path.split('/')[-1].split('.')[0]
    target = img_path.split('/')[-1].split('_')[0]
    print(f"\r\033[K{img_path}", end="")
    return grade.T, force.T, cc.T, img_name, target

# Processamento paralelo para extração de características de redes complexas com joblib
results = Parallel(n_jobs=10)(delayed(process_image)(img_path) for img_path in img_paths)

# Separar os resultados das descrições das imagens
degrees, forces, clustering, img_names, targets = zip(*results)

# Combinando graus, forças e clustering em um único conjunto de características
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
    CNFeatures = [np.concatenate([d, f, c], axis=1) for d, f, c in zip(degrees, forces, clustering)]

# Instanciando a classe FisherVectorEncoding
fisher_encoding = FisherVectorEncoding(k)

# Função para calcular os Fisher Vectors
def compute_fisher_vector(descriptor):
    print(f"\r\033[K{descriptor}", end="")
    return fisher_encoding.compute_fisher_vectors([descriptor], [descriptor])[0]

# Processamento paralelo para cálculo dos Fisher Vectors com joblib
all_fisher_vectors = Parallel(n_jobs=10)(delayed(compute_fisher_vector)(desc) for desc in CNFeatures)

# Função para separar o nome da espécie e o número da amostra
def get_species_sample_without_rotation(img_name):
    parts = img_name.split('_')
    return f"{parts[0]}_{parts[1]}"

unique_samples = sorted(set(get_species_sample_without_rotation(name) for name in img_names))

# Listas para armazenar resultados globais
overall_targets = []
svm_overall_predictions = []
svm_accuracies = []
lda_overall_predictions = []
lda_accuracies = []

# Função para processar cada amostra
def process_sample(sample, img_names, all_fisher_vectors, targets):
    print(f"\r\033[KProcessando {sample}", end="")

    # Conjunto de teste: todas as imagens que pertencem à mesma amostra
    test_indices = [i for i, name in enumerate(img_names) if get_species_sample_without_rotation(name) == sample]
    training_indices = [i for i in range(len(img_names)) if i not in test_indices]

    testing_fvs = np.array([all_fisher_vectors[i] for i in test_indices])
    training_fvs = np.array([all_fisher_vectors[i] for i in training_indices])
    train_targets = [targets[i] for i in training_indices]
    test_targets = [targets[i] for i in test_indices]

    # Verifique se o conjunto de treino é 2D: (n_amostras, n_características)
    training_fvs = training_fvs.reshape(len(training_fvs), -1)
    testing_fvs = testing_fvs.reshape(len(testing_fvs), -1)

    # Treinando o modelo SVM com os vetores de Fisher de treino
    svm = LinearSVC(random_state=42, max_iter=10000, dual='auto')
    svm.fit(training_fvs, train_targets)

    # Treinando o modelo LDA com os vetores de Fisher de treino
    lda = LinearDiscriminantAnalysis(solver='eigen', store_covariance=True, shrinkage='auto')
    lda.fit(training_fvs, train_targets)

    # Realizando predições com o modelo treinado
    prediction_svm = svm.predict(testing_fvs)
    prediction_lda = lda.predict(testing_fvs)

    # Avaliando as predições
    svm_accuracy = accuracy_score(test_targets, prediction_svm)
    lda_accuracy = accuracy_score(test_targets, prediction_lda)

    # Retornar as acurácias para cada modelo
    return svm_accuracy, lda_accuracy

# Executando o processamento em paralelo
results = Parallel(n_jobs=10)(
    delayed(process_sample)(sample, img_names, all_fisher_vectors, targets) for sample in unique_samples
)

# Separar as acurácias
svm_accuracies, lda_accuracies = zip(*results)

# Calculando e exibindo a acurácia média e o desvio padrão - SVM
svm_mean_accuracy = np.mean(svm_accuracies)
svm_std_accuracy = np.std(svm_accuracies)
print(f'SVM Mean accuracy: {svm_mean_accuracy:.4f} ± {svm_std_accuracy:.4f}')

# Calculando e exibindo a acurácia média e o desvio padrão - LDA
lda_mean_accuracy = np.mean(lda_accuracies)
lda_std_accuracy = np.std(lda_accuracies)
print(f'LDA Mean accuracy: {lda_mean_accuracy:.4f} ± {lda_std_accuracy:.4f}')

# Salvando os resultados no arquivo CSV
import csv

with open('results_mod.csv', mode='a') as csvfile:
    fieldnames = ['d', 'f', 'c', 'thre_inc', 'n_modes', 'n_fvs', 'svm_acc', 'svm_std', 'lda_acc', 'lda_std']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Escrevendo os resultados
    writer.writerow({
        'd': d_ctrl,
        'f': f_ctrl,
        'c': c_ctrl,
        'thre_inc': N,
        'n_modes': k,
        'n_fvs': len(all_fisher_vectors[0]),  # Tamanho do vetor de Fisher
        'svm_acc': f'{svm_mean_accuracy:.4f}',
        'svm_std': f'{svm_std_accuracy:.4f}',
        'lda_acc': f'{lda_mean_accuracy:.4f}',
        'lda_std': f'{lda_std_accuracy:.4f}'
    })

# Limpar as variáveis globais ao final
del degrees, forces, clustering, CNFeatures, all_fisher_vectors
gc.collect()
