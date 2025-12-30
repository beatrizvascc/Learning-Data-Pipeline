"""
Geração Automática de Dataset de Teste

Gera um dataset de primos automaticamente usando o código base (core/primos).
Permite que qualquer pessoa rode o pipeline sem arquivos externos.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math

# Adiciona o diretório core ao path para importar módulos
CORE_DIR = Path(__file__).parent.parent.parent.parent.parent / "core"
sys.path.insert(0, str(CORE_DIR))

try:
    from primos.gerador_primos import gerar_primos
except ImportError:
    # Fallback: implementação local se core não estiver disponível
    def gerar_primos(limite):
        """Crivo de Eratóstenes - fallback local."""
        crivo = [True] * (limite + 1)
        crivo[0:2] = [False, False]
        for i in range(2, int(math.isqrt(limite)) + 1):
            if crivo[i]:
                crivo[i * i: limite + 1: i] = [False] * len(range(i * i, limite + 1, i))
        return [i for i, is_prime in enumerate(crivo) if is_prime]

from .config import (
    DATA_DIR, DATASET_CSV, DATASET_TXT, RESUMO_TXT,
    NUM_PRIMOS_PADRAO, LIMITE_PRIMOS_PADRAO
)


def calcular_features(primos):
    """
    Calcula features para cada primo no dataset.
    
    Args:
        primos (list): Lista de números primos ordenados
        
    Returns:
        list: Lista de dicionários com features
    """
    dataset = []
    
    for idx in range(1, len(primos)):
        primo_atual = primos[idx]
        primo_anterior = primos[idx - 1]

        # Gap entre primos consecutivos
        gap = primo_atual - primo_anterior
        
        # Soma dos dígitos
        soma_digitos = sum(int(d) for d in str(primo_atual))
        
        # Número de bits '1' na representação binária
        bits_1_binario = bin(primo_atual).count('1')
        
        # Paridade (0=par, 1=ímpar - mas primos > 2 são sempre ímpares)
        paridade = primo_atual % 2
        
        # Densidade aproximada (usando teorema dos números primos)
        densidade = idx / primo_atual if primo_atual > 0 else 0

        dataset.append({
            'index': idx,
            'primo': primo_atual,
            'gap': gap,
            'soma_digitos': soma_digitos,
            'bits_1_binario': bits_1_binario,
            'paridade': paridade,
            'densidade': densidade
        })

    return dataset


def gerar_dataset_teste(num_primos=None, limite=None, salvar=True):
    """
    Gera um dataset de teste automaticamente.
    
    Args:
        num_primos (int, optional): Número de primos a gerar. 
                                    Se None, usa NUM_PRIMOS_PADRAO
        limite (int, optional): Limite para o Crivo de Eratóstenes.
                                Se None, usa LIMITE_PRIMOS_PADRAO
        salvar (bool): Se True, salva os arquivos
        
    Returns:
        pd.DataFrame: DataFrame com o dataset gerado
    """
    if num_primos is None:
        num_primos = NUM_PRIMOS_PADRAO
    if limite is None:
        limite = LIMITE_PRIMOS_PADRAO
    
    print(f"🔢 Gerando {num_primos} primos (limite: {limite})...")
    
    # Gera primos usando o código base
    primos = gerar_primos(limite)
    
    # Limita ao número desejado
    if len(primos) > num_primos:
        primos = primos[:num_primos]
        print(f"✅ {len(primos)} primos gerados (limitado a {num_primos})")
    else:
        print(f"✅ {len(primos)} primos gerados (todos disponíveis até {limite})")
    
    # Calcula features
    print("📊 Calculando features...")
    dataset = calcular_features(primos)
    
    # Converte para DataFrame
    df = pd.DataFrame(dataset)
    
    if salvar:
        # Salva CSV
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_CSV, index=False, sep=';')
        print(f"✅ Dataset CSV salvo em: {DATASET_CSV}")
        
        # Salva TXT legível
        salvar_txt(dataset, DATASET_TXT)
        
        # Salva resumo estatístico
        salvar_resumo(dataset, RESUMO_TXT)
    
    return df


def salvar_txt(dataset, file_path):
    """Salva dataset em formato TXT legível."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("=== DATASET DE PRIMOS ===\n\n")
        for item in dataset:
            linha = (
                f"Primo: {item['primo']}, "
                f"Gap: {item['gap']}, "
                f"Soma Dígitos: {item['soma_digitos']}, "
                f"Bits 1: {item['bits_1_binario']}, "
                f"Paridade: {item['paridade']}, "
                f"Densidade: {item['densidade']:.6f}\n"
            )
            f.write(linha)
    print(f"✅ Dataset TXT salvo em: {file_path}")


def salvar_resumo(dataset, file_path):
    """Salva resumo estatístico do dataset."""
    gaps = [d['gap'] for d in dataset]
    somas = [d['soma_digitos'] for d in dataset]
    bits = [d['bits_1_binario'] for d in dataset]

    resumo = f"""
=== RESUMO ESTATÍSTICO DOS PRIMOS ===

Total de Primos Analisados: {len(dataset)}
Maior Gap: {max(gaps)}
Menor Gap: {min(gaps)}
Média dos Gaps: {np.mean(gaps):.2f}
Mediana dos Gaps: {np.median(gaps):.2f}
Desvio Padrão dos Gaps: {np.std(gaps):.2f}

Média da Soma dos Dígitos: {np.mean(somas):.2f}
Média de Bits 1 no Binário: {np.mean(bits):.2f}

Primeiro Primo: {dataset[0]['primo']}
Último Primo: {dataset[-1]['primo']}

Distribuição de Paridade (0=par, 1=ímpar):
Par: {sum(1 for d in dataset if d['paridade'] == 0)}
Ímpar: {sum(1 for d in dataset if d['paridade'] == 1)}

Densidade no Último Primo: {dataset[-1]['densidade']:.6f}
"""

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(resumo.strip())
    
    print(f"✅ Resumo estatístico salvo em: {file_path}")


def verificar_ou_gerar_dataset():
    """
    Verifica se o dataset existe. Se não existir, gera automaticamente.
    
    Returns:
        str: Caminho do dataset (CSV)
    """
    from .config import GERAR_DATASET_AUTOMATICO, DATASET_CSV
    
    if DATASET_CSV.exists():
        print(f"✅ Dataset encontrado: {DATASET_CSV}")
        return str(DATASET_CSV)
    
    if GERAR_DATASET_AUTOMATICO:
        print("⚠️  Dataset não encontrado. Gerando automaticamente...")
        gerar_dataset_teste()
        return str(DATASET_CSV)
    else:
        raise FileNotFoundError(
            f"Dataset não encontrado em {DATASET_CSV}. "
            "Configure GERAR_DATASET_AUTOMATICO=True em config.py ou "
            "gere manualmente usando gerar_dataset_teste()."
        )


if __name__ == "__main__":
    # Gera dataset de teste quando executado diretamente
    print("=" * 60)
    print("GERADOR DE DATASET DE TESTE")
    print("=" * 60)
    
    df = gerar_dataset_teste()
    
    print("\n" + "=" * 60)
    print("✅ Dataset gerado com sucesso!")
    print(f"   Total de registros: {len(df)}")
    print(f"   Colunas: {list(df.columns)}")
    print("=" * 60)


