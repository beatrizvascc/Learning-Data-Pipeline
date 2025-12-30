"""
Gerador de Números Primos - Crivo de Eratóstenes

Implementação eficiente do Crivo de Eratóstenes para gerar primos até um limite.
"""

import math


def gerar_primos(limite):
    """
    Gera primos até o valor 'limite' com Crivo de Eratóstenes.
    
    Args:
        limite (int): Número máximo até o qual gerar primos
        
    Returns:
        list: Lista de números primos até o limite
        
    Example:
        >>> primos = gerar_primos(100)
        >>> print(primos[:10])
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    """
    if limite < 2:
        return []
    
    crivo = [True] * (limite + 1)
    crivo[0:2] = [False, False]

    for i in range(2, int(math.isqrt(limite)) + 1):
        if crivo[i]:
            crivo[i * i: limite + 1: i] = [False] * len(range(i * i, limite + 1, i))

    primos = [i for i, is_prime in enumerate(crivo) if is_prime]
    return primos


