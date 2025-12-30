"""
Análise de Gaps (Espaçamentos) entre Primos

Funções para calcular e analisar os espaçamentos entre números primos consecutivos.
"""

import numpy as np
from collections import defaultdict


def calcular_gaps(primos):
    """
    Calcula os espaçamentos (gaps) entre primos consecutivos.
    
    Args:
        primos (list): Lista de números primos ordenados
        
    Returns:
        list: Lista de gaps entre primos consecutivos
        
    Example:
        >>> primos = [2, 3, 5, 7, 11, 13]
        >>> gaps = calcular_gaps(primos)
        >>> print(gaps)
        [1, 2, 2, 4, 2]
    """
    if len(primos) < 2:
        return []
    
    gaps = []
    for i in range(1, len(primos)):
        gaps.append(primos[i] - primos[i - 1])
    return gaps


def calcular_gaps_por_digito_final(primos):
    """
    Calcula os gaps agrupados por dígito final do primo anterior.
    
    Args:
        primos (list): Lista de números primos ordenados
        
    Returns:
        dict: Dicionário {dígito_final: {gap: frequência}}
        
    Example:
        >>> primos = [2, 3, 5, 7, 11, 13, 17, 19]
        >>> gaps_por_digito = calcular_gaps_por_digito_final(primos)
    """
    gaps_por_digito = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(primos) - 1):
        primo_atual = primos[i]
        proximo_primo = primos[i + 1]
        gap = proximo_primo - primo_atual
        digito_final = primo_atual % 10  # Pega o último dígito
        gaps_por_digito[digito_final][gap] += 1
    
    return gaps_por_digito


def analisar_gaps(primos):
    """
    Análise estatística dos gaps entre primos.
    
    Args:
        primos (list): Lista de números primos ordenados
        
    Returns:
        dict: Estatísticas dos gaps (média, mediana, máximo, etc.)
    """
    gaps = calcular_gaps(primos)
    
    if not gaps:
        return {}
    
    gaps_array = np.array(gaps)
    
    return {
        'total': len(gaps),
        'media': float(np.mean(gaps_array)),
        'mediana': float(np.median(gaps_array)),
        'desvio_padrao': float(np.std(gaps_array)),
        'minimo': int(np.min(gaps_array)),
        'maximo': int(np.max(gaps_array)),
        'gaps_unicos': len(set(gaps))
    }


