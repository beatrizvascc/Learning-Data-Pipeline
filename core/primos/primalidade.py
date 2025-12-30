"""
Testes de Primalidade

Implementações de algoritmos para verificar se um número é primo.
"""

import random


def miller_rabin(n, k=10):
    """
    Teste de primalidade de Miller-Rabin (probabilístico).
    
    Algoritmo probabilístico que determina se um número é provavelmente primo.
    Para k=10, a probabilidade de erro é menor que 1 em 1 milhão.
    
    Args:
        n (int): Número a ser testado
        k (int): Número de iterações (padrão: 10)
        
    Returns:
        bool: True se n é provavelmente primo, False se é composto
        
    Example:
        >>> miller_rabin(17)
        True
        >>> miller_rabin(100)
        False
    """
    if n < 2:
        return False
    if n in [2, 3]:
        return True
    if n % 2 == 0:
        return False

    # Escrevendo n - 1 como 2^s * d, com d ímpar
    s, d = 0, n - 1
    while d % 2 == 0:
        s, d = s + 1, d // 2

    def verifica_teste(a, d, n, s):
        """Verifica se o número passa no teste de Miller-Rabin para a base a."""
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return True
        return False

    # Teste com bases aleatórias
    for _ in range(k):
        a = random.randrange(2, n - 1)
        if not verifica_teste(a, d, n, s):
            return False

    return True


def eh_primo(n):
    """
    Verifica se um número é primo usando método simples (para números pequenos).
    
    Para números grandes, use miller_rabin().
    
    Args:
        n (int): Número a ser testado
        
    Returns:
        bool: True se n é primo, False caso contrário
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


