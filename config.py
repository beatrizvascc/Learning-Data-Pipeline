"""
Configurações do Pipeline de Análise de Primos

Centraliza todas as configurações de caminhos e parâmetros.
Usa caminhos relativos para portabilidade.
"""

import os
from pathlib import Path

# ======================
# CAMINHOS BASE (Dinâmicos)
# ======================

# Diretório base do projeto (raiz do codebase)
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent

# Diretório de dados (relativo ao projeto)
DATA_DIR = BASE_DIR / "data" / "primos"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Diretório de resultados (relativo ao projeto)
RESULTS_DIR = BASE_DIR / "data" / "resultados"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_IMG_DIR = RESULTS_DIR / "imagens"
RESULTS_IMG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_REL_DIR = RESULTS_DIR / "relatorios"
RESULTS_REL_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# ARQUIVOS DE DADOS
# ======================

# Dataset principal (CSV)
DATASET_CSV = DATA_DIR / "dataset_primos.csv"

# Arquivo de primos (TXT) - opcional, usado apenas se fornecido
PRIMOS_FILE = DATA_DIR / "primos.txt"

# Dataset em formato TXT (legível)
DATASET_TXT = DATA_DIR / "dataset_primos.txt"

# Resumo estatístico
RESUMO_TXT = DATA_DIR / "resumo_analise_primos.txt"

# ======================
# ARQUIVOS DE RESULTADOS
# ======================

# Imagens geradas
FOURIER_IMG = RESULTS_IMG_DIR / "fourier.png"
WAVELET_IMG = RESULTS_IMG_DIR / "wavelet.png"
PCA_IMG = RESULTS_IMG_DIR / "pca.png"
AUTOENCODER_IMG = RESULTS_IMG_DIR / "autoencoder_erro.png"
GNN_IMG = RESULTS_IMG_DIR / "grafo_gnn.png"

# Relatórios
RELATORIO_TXT = RESULTS_REL_DIR / "relatorio_primos.txt"
RELATORIO_AVANCADO_TXT = RESULTS_REL_DIR / "relatorio_primos_avancado.txt"
RELATORIO_AVANCADO_CSV = RESULTS_REL_DIR / "relatorio_primos_avancado.csv"

# ======================
# CONFIGURAÇÕES DE ANÁLISE
# ======================

# Controle de faixa no DataFrame
RANGE_INICIO = 0
RANGE_FIM = None  # None = carregar tudo

# Ativa / desativa os módulos de análise
RODAR_FOURIER = True
RODAR_WAVELET = True
RODAR_PCA = True
RODAR_AUTOENCODER = True
RODAR_GNN = True

# ======================
# CONFIGURAÇÕES DE GERAÇÃO DE DATASET
# ======================

# Parâmetros para geração automática de dataset de teste
GERAR_DATASET_AUTOMATICO = True  # Se True, gera dataset se não existir
NUM_PRIMOS_PADRAO = 10000  # Número de primos para dataset de teste
LIMITE_PRIMOS_PADRAO = 100000  # Limite para gerar primos (Crivo de Eratóstenes)

# ======================
# CONFIGURAÇÕES DE MODELO
# ======================

# Autoencoder
AUTOENCODER_EPOCHS = 300
AUTOENCODER_BATCH_SIZE = 256
AUTOENCODER_VERBOSE = 1

# Wavelet
WAVELET_TYPE = 'db1'
WAVELET_LEVEL = 6

# PCA
PCA_N_COMPONENTS = 2

# ======================
# FUNÇÕES AUXILIARES
# ======================

def get_dataset_path():
    """Retorna o caminho do dataset CSV."""
    return str(DATASET_CSV)

def get_primos_path():
    """Retorna o caminho do arquivo de primos."""
    return str(PRIMOS_FILE)

def get_results_img_dir():
    """Retorna o diretório de imagens de resultados."""
    return str(RESULTS_IMG_DIR)

def get_results_rel_dir():
    """Retorna o diretório de relatórios."""
    return str(RESULTS_REL_DIR)

def criar_diretorios():
    """Cria todos os diretórios necessários."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_REL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Diretórios criados/verificados:")
    print(f"   - Dados: {DATA_DIR}")
    print(f"   - Resultados: {RESULTS_DIR}")


