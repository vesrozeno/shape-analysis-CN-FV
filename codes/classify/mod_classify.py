import pickle
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import os
import psutil
import time

# Variáveis para determinar os nomes de arquivos
results_csv = "results/usp_scaled.csv"  # Arquivo CSV para salvar os resultados
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

    # Obter os caminhos dos arquivos de treino e teste
    train_pkl = "pkl/usp_orig.pkl"
    test_pkl = "pkl/usp_sca.pkl"

    # Lendo combinações já processadas
    processed_combinations = read_processed_combinations(results_csv)
    print(f"Número de combinações já processadas: {len(processed_combinations)}")

    print("Carregando Fisher Vectors e rótulos de treino...")
    # Carregar os Fisher Vectors e os rótulos de treino
    with open(train_pkl, "rb") as f:
        train_data = pickle.load(f)
    train_fisher_vectors_dict = train_data["fisher_vectors"]
    train_targets = train_data["targets"]

    print("Carregando Fisher Vectors e rótulos de teste...")
    # Carregar os Fisher Vectors e os rótulos de teste
    with open(test_pkl, "rb") as f:
        test_data = pickle.load(f)
    test_fisher_vectors_dict = test_data["fisher_vectors"]
    test_targets = test_data["targets"]

    print("Fisher Vectors e rótulos carregados com sucesso.\n")

    # Validar para cada combinação de parâmetros, sem barra de progresso
    for params, train_fisher_vectors in train_fisher_vectors_dict.items():
        test_fisher_vectors = test_fisher_vectors_dict[params]  # Certifique-se de que os parâmetros são iguais
        process_combination(params, train_fisher_vectors, train_targets, test_fisher_vectors, test_targets, processed_combinations)

def process_combination(params, train_fisher_vectors, train_targets, test_fisher_vectors, test_targets, processed_combinations):
    d_ctrl, f_ctrl, c_ctrl, k, N = params

    # Verificando se a combinação já foi processada
    if (d_ctrl, f_ctrl, c_ctrl, k, N) not in processed_combinations:
        # Verifique se os limites de recursos estão dentro do aceitável antes de processar
        while not check_system_resources():
            print("Aguardando recursos suficientes para continuar a execução...")
            time.sleep(5)  # Aguarde 5 segundos antes de verificar novamente

        # Executar a validação do modelo
        validate_model(train_fisher_vectors, train_targets, test_fisher_vectors, test_targets, d_ctrl, f_ctrl, c_ctrl, k, N)

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

def validate_model(train_fisher_vectors, train_targets, test_fisher_vectors, test_targets, d_ctrl, f_ctrl, c_ctrl, k, N):
    svm_accuracies = []
    lda_accuracies = []

    # Treinando SVM
    svm = LinearSVC(random_state=42, max_iter=10000, dual="auto")
    svm.fit(train_fisher_vectors, train_targets)
    predictions_svm = svm.predict(test_fisher_vectors)
    svm_accuracy = accuracy_score(test_targets, predictions_svm)
    svm_accuracies.append(svm_accuracy)

    # Treinando LDA
    lda = LinearDiscriminantAnalysis(solver="eigen", store_covariance=True, shrinkage="auto")
    lda.fit(train_fisher_vectors, train_targets)
    predictions_lda = lda.predict(test_fisher_vectors)
    lda_accuracy = accuracy_score(test_targets, predictions_lda)
    lda_accuracies.append(lda_accuracy)

    svm_mean_accuracy = np.mean(svm_accuracies)
    svm_std_accuracy = np.std(svm_accuracies)
    lda_mean_accuracy = np.mean(lda_accuracies)
    lda_std_accuracy = np.std(lda_accuracies)

    save_results(d_ctrl, f_ctrl, c_ctrl, N, k, len(train_fisher_vectors[0]), svm_mean_accuracy, svm_std_accuracy, lda_mean_accuracy, lda_std_accuracy)

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

if __name__ == "__main__":
    main()
