import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import os
from joblib import Parallel, delayed
import psutil
import time

# Variáveis para determinar os nomes de arquivos
fisher_vectors_pkl = "pkl/ethF.pkl"  # Arquivo pkl com os Fisher Vectors e rótulos
results_csv = "results/results_eth.csv"  # Arquivo CSV para salvar os resultados
n_jobs = 8  # Número de núcleos para usar (-1 usa todos os núcleos disponíveis)
memory_limit_mb = 4096  # Limite de memória em MB (4GB por padrão)
cpu_limit_percent = 80  # Limite de uso da CPU em porcentagem

def main():
    global n_jobs, memory_limit_mb, cpu_limit_percent
    
    # Permitir que o usuário defina o número de núcleos a serem utilizados
    try:
        n_jobs = int(input("Digite o número de núcleos a serem utilizados (ou -1 para usar todos os núcleos disponíveis): "))
    except ValueError:
        print("Entrada inválida. Usando todos os núcleos disponíveis.")
        n_jobs = 8
    
    # Permitir que o usuário defina o limite de memória
    try:
        memory_limit_mb = int(input(f"Digite o limite de memória em MB (padrão: {memory_limit_mb} MB): "))
    except ValueError:
        print(f"Entrada inválida. Usando o limite padrão de {memory_limit_mb} MB.")
    
    # Permitir que o usuário defina o limite de uso da CPU
    try:
        cpu_limit_percent = int(input(f"Digite o limite máximo de uso da CPU em porcentagem (padrão: {cpu_limit_percent}%): "))
    except ValueError:
        print(f"Entrada inválida. Usando o limite padrão de {cpu_limit_percent}%.")
    
    # Lendo combinações já processadas
    processed_combinations = read_processed_combinations(results_csv)
    print(f"Número de combinações já processadas: {len(processed_combinations)}")

    print("Carregando Fisher Vectors e rótulos...")
    # Carregar os Fisher Vectors e os rótulos
    with open(fisher_vectors_pkl, "rb") as f:
        data = pickle.load(f)
    fisher_vectors_dict = data["fisher_vectors"]
    targets = data["targets"]
    samples = data["samples"]

    print("Fisher Vectors e rótulos carregados com sucesso.\n")
    print(f"Length of fisher_vectors: {len(fisher_vectors_dict)}")
    print(f"Length of targets: {len(targets)}")
    print(f"Length of samples: {len(samples)}")

    # Executar a classificação para cada combinação de parâmetros sem barra de progresso
    Parallel(n_jobs=get_adjusted_n_jobs())(
        delayed(process_combination)(params, fisher_vectors, targets, samples, processed_combinations) 
        for params, fisher_vectors in fisher_vectors_dict.items()
    )

def process_combination(params, fisher_vectors, targets, samples, processed_combinations):
    d_ctrl, f_ctrl, c_ctrl, k, N = params

    # Verificando se a combinação já foi processada
    if (d_ctrl, f_ctrl, c_ctrl, k, N) not in processed_combinations:
        # Verifique se os limites de recursos estão dentro do aceitável antes de processar
        while not check_system_resources():
            print("Aguardando recursos suficientes para continuar a execução...")
            time.sleep(5)  # Aguarde 5 segundos antes de verificar novamente

        # Executar a validação do modelo
        validate_model(fisher_vectors, targets, samples, d_ctrl, f_ctrl, c_ctrl, k, N)

def read_processed_combinations(csv_file):
    processed_combinations = set()
    if os.path.exists(csv_file):
        try:
            with open(csv_file, mode='r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    d_ctrl = int(row['d'])
                    f_ctrl = int(row['f'])
                    c_ctrl = int(row['c'])
                    k = int(row['n_modes'])
                    N = int(row['thre_inc'])
                    processed_combinations.add((d_ctrl, f_ctrl, c_ctrl, k, N))
        except Exception as e:
            print(f"Erro ao ler o arquivo CSV: {e}")
    return processed_combinations

def validate_model(fisher_vectors, targets, samples, d_ctrl, f_ctrl, c_ctrl, k, N):

    svm_accuracies = []
    lda_accuracies = []

    # Itera sobre todas as amostras, usando cada uma delas como conjunto de teste
    unique_samples = set(samples)
    for sample in unique_samples:
        print(f"\r\033[KProcessando amostra: {sample}", end="")

        # Conjunto de teste: todas as imagens que pertencem à mesma amostra
        test_indices = [i for i, s in enumerate(samples) if s == sample]
        training_indices = [i for i in range(len(samples)) if i not in test_indices]

        # Separando vetores de treino e teste
        train_fvs = [fisher_vectors[i] for i in training_indices]
        test_fvs = [fisher_vectors[i] for i in test_indices]
        train_targets = [targets[i] for i in training_indices]
        test_targets = [targets[i] for i in test_indices]

        # Treinando SVM
        svm = LinearSVC(random_state=42, max_iter=10000, dual="auto")
        svm.fit(train_fvs, train_targets)
        predictions_svm = svm.predict(test_fvs)
        svm_accuracy = accuracy_score(test_targets, predictions_svm)
        svm_accuracies.append(svm_accuracy)

        # Treinando LDA
        lda = LinearDiscriminantAnalysis(solver="eigen", store_covariance=True, shrinkage="auto")
        lda.fit(train_fvs, train_targets)
        predictions_lda = lda.predict(test_fvs)
        lda_accuracy = accuracy_score(test_targets, predictions_lda)
        lda_accuracies.append(lda_accuracy)

    # Calculando as métricas de desempenho
    svm_mean_accuracy = np.mean(svm_accuracies)
    svm_std_accuracy = np.std(svm_accuracies)
    lda_mean_accuracy = np.mean(lda_accuracies)
    lda_std_accuracy = np.std(lda_accuracies)

    save_results(d_ctrl, f_ctrl, c_ctrl, N, k, len(fisher_vectors[0]), svm_mean_accuracy, svm_std_accuracy, lda_mean_accuracy, lda_std_accuracy)

def validate_model_leave_one_out(fisher_vectors, targets, d_ctrl, f_ctrl, c_ctrl, k, N):
    loo = LeaveOneOut()

    svm_accuracies = []
    lda_accuracies = []

    # Realiza o Leave-One-Out
    for train_index, test_index in loo.split(fisher_vectors):
        train_fvs = [fisher_vectors[i] for i in train_index]
        test_fv = [fisher_vectors[i] for i in test_index]  # Apenas uma amostra de teste
        train_targets = [targets[i] for i in train_index]
        test_target = [targets[i] for i in test_index]  # Apenas um alvo de teste

        # Treinando SVM
        svm = LinearSVC(random_state=42, max_iter=10000, dual="auto")
        svm.fit(train_fvs, train_targets)
        predictions_svm = svm.predict(test_fv)
        svm_accuracy = accuracy_score(test_target, predictions_svm)
        svm_accuracies.append(svm_accuracy)

        # Treinando LDA
        lda = LinearDiscriminantAnalysis(solver="eigen", store_covariance=True, shrinkage="auto")
        lda.fit(train_fvs, train_targets)
        predictions_lda = lda.predict(test_fv)
        lda_accuracy = accuracy_score(test_target, predictions_lda)
        lda_accuracies.append(lda_accuracy)

    # Calculando as métricas de desempenho
    svm_mean_accuracy = np.mean(svm_accuracies)
    svm_std_accuracy = np.std(svm_accuracies)
    lda_mean_accuracy = np.mean(lda_accuracies)
    lda_std_accuracy = np.std(lda_accuracies)

    save_results(d_ctrl, f_ctrl, c_ctrl, N, k, len(fisher_vectors[0]), svm_mean_accuracy, svm_std_accuracy, lda_mean_accuracy, lda_std_accuracy)
    
def save_results(d_ctrl, f_ctrl, c_ctrl, N, k, num_fisher_vectors, svm_mean_accuracy, svm_std_accuracy, lda_mean_accuracy, lda_std_accuracy):
    with open(results_csv, mode="a", newline="") as csvfile:
        fieldnames = ["d", "f", "c", "thre_inc", "n_modes", "n_fvs", "svm_acc", "svm_std", "lda_acc", "lda_std"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Verificar se o arquivo está vazio para escrever o cabeçalho
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({
            "d": d_ctrl,
            "f": f_ctrl,
            "c": c_ctrl,
            "thre_inc": N,
            "n_modes": k,
            "n_fvs": num_fisher_vectors,
            "svm_acc": f"{svm_mean_accuracy:.4f}",
            "svm_std": f"{svm_std_accuracy:.4f}",
            "lda_acc": f"{lda_mean_accuracy:.4f}",
            "lda_std": f"{lda_std_accuracy:.4f}",
        })

def check_system_resources():
    """
    Verifica se o sistema tem recursos suficientes para continuar a execução.
    Retorna True se os recursos estiverem dentro dos limites definidos, caso contrário, False.
    """
    memory_info = psutil.virtual_memory()
    memory_used_mb = memory_info.used / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=1)

    # Construa a string de status de recursos
    resource_status = (
        f"Uso de memória: {memory_used_mb:.2f} MB, Limite: {memory_limit_mb} MB | "
        f"Uso de CPU: {cpu_percent:.2f}%, Limite: {cpu_limit_percent}%"
    )

    # Imprima a string de status de recursos com retorno de carro e limpeza de linha
    print(f"\r{resource_status}\033[K", end="")

    # Verificar se os limites de memória e CPU foram excedidos
    if memory_used_mb > memory_limit_mb or cpu_percent > cpu_limit_percent:
        return False
    return True

def get_adjusted_n_jobs():
    """
    Ajusta o número de jobs (núcleos) usados com base no uso atual da CPU.
    Se o uso da CPU estiver perto do limite, reduz o número de jobs.
    """
    cpu_percent = psutil.cpu_percent(interval=1)
    
    if cpu_percent > cpu_limit_percent:
        return max(1, n_jobs // 2)  # Reduz para metade ou 1 job se estiver perto do limite
    return n_jobs

if __name__ == "__main__":
    main()
