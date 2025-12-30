"""
Pipeline Completo de Análise de Primos

Análise avançada de números primos usando múltiplas técnicas:
- Transformada de Fourier (FFT)
- Análise Wavelet
- Análise de Componentes Principais (PCA)
- Autoencoder (Deep Learning)
- Graph Neural Networks (GNN)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import tensorflow as tf
from sklearn.decomposition import PCA
import networkx as nx
from tensorflow.keras import layers, models

# Importa configurações dinâmicas
from .config import (
    get_dataset_path, get_results_img_dir, get_results_rel_dir,
    RANGE_INICIO, RANGE_FIM,
    RODAR_FOURIER, RODAR_WAVELET, RODAR_PCA, RODAR_AUTOENCODER, RODAR_GNN,
    FOURIER_IMG, WAVELET_IMG, PCA_IMG, AUTOENCODER_IMG, GNN_IMG,
    RELATORIO_TXT,
    AUTOENCODER_EPOCHS, AUTOENCODER_BATCH_SIZE, AUTOENCODER_VERBOSE,
    WAVELET_TYPE, WAVELET_LEVEL, PCA_N_COMPONENTS,
    criar_diretorios
)

# Importa gerador de dataset
from .gerar_dataset import verificar_ou_gerar_dataset


# ======================
# FUNÇÕES AUXILIARES
# ======================

def carregar_dataset(file_path=None, inicio=None, fim=None):
    """
    Carrega o dataset de primos.
    
    Args:
        file_path (str, optional): Caminho do arquivo CSV. 
                                   Se None, usa o caminho padrão ou gera automaticamente
        inicio (int, optional): Índice inicial para filtrar dados
        fim (int, optional): Índice final para filtrar dados
        
    Returns:
        pd.DataFrame: DataFrame com os dados dos primos
    """
    if file_path is None:
        # Verifica ou gera dataset automaticamente
        file_path = verificar_ou_gerar_dataset()
    
    try:
        df = pd.read_csv(file_path, sep=';')
        print(f"✅ Dataset carregado: {len(df)} registros de {file_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset não encontrado em {file_path}. "
            "Execute gerar_dataset.py para criar um dataset de teste."
        )
    
    if inicio is not None and fim is not None:
        df = df.iloc[inicio:fim]
        print(f"➡️  Intervalo aplicado: {inicio} a {fim} ({len(df)} registros)")
    elif inicio is not None:
        df = df.iloc[inicio:]
        print(f"➡️  Intervalo aplicado: a partir de {inicio} ({len(df)} registros)")
    elif fim is not None:
        df = df.iloc[:fim]
        print(f"➡️  Intervalo aplicado: até {fim} ({len(df)} registros)")

    return df


# ======================
# ANÁLISES
# ======================

def analise_fourier(gaps, salvar_grafico=True):
    """
    Análise de Fourier (FFT) dos gaps entre primos.
    
    Args:
        gaps (array-like): Array de gaps entre primos
        salvar_grafico (bool): Se True, salva o gráfico
        
    Returns:
        str: Mensagem de conclusão
    """
    fft = np.fft.fft(gaps)
    freq = np.fft.fftfreq(len(gaps))

    plt.figure(figsize=(10, 5))
    plt.plot(freq[:len(freq)//2], np.abs(fft[:len(fft)//2]))
    plt.title('Transformada de Fourier dos Gaps dos Primos')
    plt.xlabel('Frequência')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    
    if salvar_grafico:
        plt.savefig(FOURIER_IMG, dpi=150, bbox_inches='tight')
        print(f"   📊 Gráfico salvo: {FOURIER_IMG}")
    
    plt.close()

    return "✅ Fourier concluída."


def analise_wavelet(gaps, salvar_grafico=True):
    """
    Análise Wavelet dos gaps entre primos.
    
    Args:
        gaps (array-like): Array de gaps entre primos
        salvar_grafico (bool): Se True, salva o gráfico
        
    Returns:
        str: Mensagem de conclusão
    """
    coeficientes = pywt.wavedec(gaps, WAVELET_TYPE, level=WAVELET_LEVEL)

    plt.figure(figsize=(12, 8))
    for i, c in enumerate(coeficientes):
        plt.subplot(len(coeficientes), 1, i + 1)
        plt.plot(c)
        plt.title(f'Coeficiente Wavelet Nível {i + 1}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if salvar_grafico:
        plt.savefig(WAVELET_IMG, dpi=150, bbox_inches='tight')
        print(f"   📊 Gráfico salvo: {WAVELET_IMG}")
    
    plt.close()

    return "✅ Wavelet concluída."


def analise_pca(df, salvar_grafico=True):
    """
    Análise de Componentes Principais (PCA) dos dados.
    
    Args:
        df (pd.DataFrame): DataFrame com features dos primos
        salvar_grafico (bool): Se True, salva o gráfico
        
    Returns:
        str: Mensagem de conclusão
    """
    # Verifica se as colunas necessárias existem
    colunas_necessarias = ['gap', 'soma_digitos', 'bits_1_binario', 'paridade', 'densidade']
    colunas_disponiveis = [col for col in colunas_necessarias if col in df.columns]
    
    if len(colunas_disponiveis) < 2:
        return "⚠️ PCA pulada: colunas insuficientes no dataset"
    
    features = df[colunas_disponiveis].values
    pca = PCA(n_components=PCA_N_COMPONENTS)
    componentes = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    plt.scatter(componentes[:, 0], componentes[:, 1], s=1, color='green', alpha=0.6)
    plt.title('Análise de Componentes Principais (PCA) dos Primos')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.2%} variância)')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.2%} variância)')
    plt.grid(True, alpha=0.3)
    
    if salvar_grafico:
        plt.savefig(PCA_IMG, dpi=150, bbox_inches='tight')
        print(f"   📊 Gráfico salvo: {PCA_IMG}")
    
    plt.close()

    return f"✅ PCA concluída. Variância explicada: {sum(pca.explained_variance_ratio_):.2%}"


def analise_autoencoder(gaps, salvar_grafico=True):
    """
    Autoencoder para detectar padrões e anomalias nos gaps.
    
    Args:
        gaps (array-like): Array de gaps entre primos
        salvar_grafico (bool): Se True, salva o gráfico
        
    Returns:
        str: Mensagem de conclusão com erro médio
    """
    # 1. Prepara os dados para o TensorFlow (2D array)
    X = np.array(gaps).reshape(-1, 1)

    # 2. Normalização para ficar entre 0 e 1
    X_max = np.max(X)
    if X_max == 0:
        return "⚠️ Autoencoder pulado: dados inválidos (todos zeros)"
    
    X_normalized = X / X_max

    # 3. Criação do modelo
    model = models.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # 4. Treinamento
    print(f"   🧠 Treinando autoencoder ({AUTOENCODER_EPOCHS} épocas)...")
    history = model.fit(
        X_normalized, X_normalized, 
        epochs=AUTOENCODER_EPOCHS, 
        batch_size=AUTOENCODER_BATCH_SIZE, 
        verbose=AUTOENCODER_VERBOSE
    )

    # 5. Reconstrução e cálculo do erro
    reconstruido = model.predict(X_normalized, batch_size=32768, verbose=0)
    erro = np.abs(X_normalized - reconstruido)
    erro_medio = np.mean(erro)

    # 6. Plot do erro
    plt.figure(figsize=(10, 5))
    plt.plot(erro, alpha=0.7)
    plt.title('Erro de Reconstrução (Autoencoder) - Anomalias nos Gaps')
    plt.xlabel('Amostra')
    plt.ylabel('Erro de Reconstrução')
    plt.grid(True, alpha=0.3)
    
    if salvar_grafico:
        plt.savefig(AUTOENCODER_IMG, dpi=150, bbox_inches='tight')
        print(f"   📊 Gráfico salvo: {AUTOENCODER_IMG}")
    
    plt.close()

    return f"✅ Autoencoder concluído. Erro médio: {erro_medio:.6f}"


def analise_gnn(df, salvar_grafico=True):
    """
    Cria grafo dos gaps entre primos (Graph Neural Network base).
    
    Args:
        df (pd.DataFrame): DataFrame com gaps
        salvar_grafico (bool): Se True, salva o gráfico
        
    Returns:
        str: Mensagem de conclusão
    """
    if 'gap' not in df.columns:
        return "⚠️ GNN pulado: coluna 'gap' não encontrada"
    
    gaps = df['gap'].values
    G = nx.Graph()

    # Limita o tamanho do grafo para visualização (primeiros 1000 nós)
    limite_visualizacao = min(1000, len(gaps) - 1)
    
    for i in range(limite_visualizacao):
        G.add_edge(i, i + 1, weight=gaps[i])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=0.3, iterations=50)
    nx.draw(G, pos, node_size=10, edge_color='gray', alpha=0.6, width=0.5)
    plt.title(f'Grafo dos Gaps dos Primos (primeiros {limite_visualizacao} nós)')
    
    if salvar_grafico:
        plt.savefig(GNN_IMG, dpi=150, bbox_inches='tight')
        print(f"   📊 Gráfico salvo: {GNN_IMG}")
    
    plt.close()

    return f"✅ GNN (grafo) gerado com {G.number_of_nodes()} nós."


# ======================
# RELATÓRIO FINAL
# ======================

def gerar_relatorio(conteudo, nome_arquivo=None):
    """
    Gera relatório final com todos os resultados.
    
    Args:
        conteudo (list): Lista de strings com resultados
        nome_arquivo (str, optional): Caminho do arquivo. Se None, usa padrão
    """
    if nome_arquivo is None:
        nome_arquivo = RELATORIO_TXT
    
    with open(nome_arquivo, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("RELATÓRIO COMPLETO - ANÁLISE DE PRIMOS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for item in conteudo:
            f.write(f"{item}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("FIM DO RELATÓRIO\n")
        f.write("=" * 60 + "\n")
    
    print(f'📄 Relatório salvo: {nome_arquivo}')


# ======================
# PIPELINE PRINCIPAL
# ======================

def pipeline_primos(file_path=None, inicio=None, fim=None):
    """
    Pipeline principal de análise de primos.
    
    Args:
        file_path (str, optional): Caminho do dataset CSV. Se None, usa padrão ou gera automaticamente
        inicio (int, optional): Índice inicial para filtrar dados
        fim (int, optional): Índice final para filtrar dados
    """
    print("=" * 60)
    print("PIPELINE COMPLETO DE ANÁLISE DE PRIMOS")
    print("=" * 60)
    
    # Cria diretórios necessários
    criar_diretorios()
    
    # Carrega dataset
    print("\n📂 Carregando dataset...")
    df = carregar_dataset(file_path, inicio or RANGE_INICIO, fim or RANGE_FIM)
    
    if 'gap' not in df.columns:
        raise ValueError("Dataset deve conter coluna 'gap'")
    
    gaps = df['gap'].values
    print(f"   Total de gaps analisados: {len(gaps)}")
    
    # Executa análises
    print("\n🔬 Executando análises...")
    resultados = []
    
    if RODAR_FOURIER:
        print("   📊 Análise de Fourier...")
        resultados.append(analise_fourier(gaps))

    if RODAR_WAVELET:
        print("   📊 Análise Wavelet...")
        resultados.append(analise_wavelet(gaps))

    if RODAR_PCA:
        print("   📊 Análise PCA...")
        resultados.append(analise_pca(df))

    if RODAR_AUTOENCODER:
        print("   🧠 Autoencoder...")
        resultados.append(analise_autoencoder(gaps))

    if RODAR_GNN:
        print("   📊 Graph Neural Network...")
        resultados.append(analise_gnn(df))

    # Gera relatório
    print("\n📄 Gerando relatório...")
    gerar_relatorio(resultados)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)


# ======================
# EXECUÇÃO
# ======================

if __name__ == "__main__":
    pipeline_primos()
