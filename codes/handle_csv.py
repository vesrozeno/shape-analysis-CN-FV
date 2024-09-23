import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = input("Dataset: ")
legenda = input("Legenda: ")

# Ler o arquivo CSV
file_path = "results/results_{}.csv".format(dataset)  # Coloque o caminho correto do seu arquivo CSV

def main():
    sair = 0
    df = pd.read_csv(file_path)

    # Filtrar os dados onde d, f e c são iguais a 1
    df_filtered = df[(df['d'] == 1) & (df['f'] == 0) & (df['c'] == 0)]

    # Verificações para thre_inc e n_modes
    thre_inc_values = [10, 20, 30, 40, 50, 60, 70]
    n_modes_values = [4, 6, 8, 10, 14, 18, 22]

    # Filtrar as colunas 'thre_inc' e 'n_modes' com os valores desejados
    df_filtered = df_filtered[df_filtered['thre_inc'].isin(thre_inc_values)]
    df_filtered = df_filtered[df_filtered['n_modes'].isin(n_modes_values)]

    # Verificar se há dados válidos após o filtro
    if df_filtered.empty:
        print("Nenhum dado disponível para os valores especificados de thre_inc e n_modes.")
    else:
        while sair == 0:
            combine = int(input("Combinar: (1/0): "))
            if combine == 0:
                classifier = int(input("Classificador: (1 - SVM/0 - LDA): "))
            else:
                classifier = 0

            if combine:
                combined(df_filtered)
            elif classifier:
                svm(df_filtered)
            else:
                lda(df_filtered)
            sair = int(input("Sair? (1/0): "))

def combined(df_filtered):
    # Criar uma nova coluna 'avg_acc' que é a média entre 'svm_acc' e 'lda_acc'
        df_filtered['avg_acc'] = df_filtered[['svm_acc', 'lda_acc']].mean(axis=1)

        # Criar uma tabela de texto que combina acc_svm e acc_lda
        df_filtered['combined_acc'] = df_filtered.apply(
            lambda row: f"SVM: {row['svm_acc']:.2f}\nLDA: {row['lda_acc']:.2f}", axis=1)

        # Criar tabela pivô para o heatmap (usando avg_acc para colorir o heatmap)
        heatmap_data_avg = df_filtered.pivot(index="n_modes", columns="thre_inc", values="avg_acc")

        # Criar tabela manual para exibir os textos (SVM e LDA)
        heatmap_data_text = df_filtered.pivot(index="n_modes", columns="thre_inc", values="combined_acc")

        # Criar o heatmap, mas usando avg_acc para a coloração
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data_avg, annot=heatmap_data_text.values, fmt='', cmap="YlGnBu", 
                        cbar_kws={'label': 'Acc Média (SVM & LDA)'})

        # Ajustar título e rótulos dos eixos
        plt.title("Heatmap SVM & LDA Acc Média - (Grau) - {}".format(legenda))
        plt.xlabel("Limiares")
        plt.ylabel("K Gaussianas")
        plt.savefig('/mnt/d/{}/comb_hm_{}.png'.format(dataset, dataset))

def svm(df_filtered):
    # Pivotar os dados para criar o formato de heatmap
    heatmap_data = df_filtered.pivot_table(index="n_modes", columns="thre_inc", values="svm_acc")

    # Criar o heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'SVM Acc'})
    plt.title("Heatmap SVM Acc - (Grau) - {}".format(legenda))
    plt.xlabel("Limiares")
    plt.ylabel("K Gaussianas")
    plt.savefig('/mnt/d/{}/svm_hm_{}.png'.format(dataset, dataset))

def lda(df_filtered):
    # Pivotar os dados para criar o formato de heatmap
    heatmap_data = df_filtered.pivot_table(index="n_modes", columns="thre_inc", values="lda_acc")

    # Criar o heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'LDA Acc'})
    plt.title("Heatmap LDA Acc - (Grau) - {}".format(legenda))
    plt.xlabel("Limiares")
    plt.ylabel("K Gaussianas")
    plt.savefig('/mnt/d/{}/lda_hm_{}.png'.format(dataset, dataset))

if __name__ == '__main__':
     
     main()