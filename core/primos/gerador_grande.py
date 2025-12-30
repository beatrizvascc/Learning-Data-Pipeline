"""
Gerador de Números Primos Grandes

Usa gmpy2 (GMP) para gerar primos muito grandes (1500+ dígitos).
"""

import gmpy2
from gmpy2 import mpz, random_state
import time
import os


def carregar_primos_gerados(arquivo_registro="primos_gerados.txt"):
    """
    Carrega os números primos já gerados a partir do arquivo.
    
    Args:
        arquivo_registro (str): Caminho do arquivo de registro
        
    Returns:
        set: Conjunto de primos já gerados
    """
    if not os.path.exists(arquivo_registro):
        return set()
    with open(arquivo_registro, "r") as arquivo:
        return set(mpz(linha.strip()) for linha in arquivo)


def salvar_primo_gerado(primo, arquivo_registro="primos_gerados.txt"):
    """
    Salva um novo número primo no arquivo de registro.
    
    Args:
        primo: Número primo a ser salvo
        arquivo_registro (str): Caminho do arquivo de registro
    """
    with open(arquivo_registro, "a") as arquivo:
        arquivo.write(f"{primo}\n")


def gerar_primo_grande(digitos=1500, arquivo_registro="primos_gerados.txt"):
    """
    Gera um número primo grande que ainda não foi gerado.
    
    Args:
        digitos (int): Número de dígitos do primo (padrão: 1500)
        arquivo_registro (str): Caminho do arquivo de registro
        
    Returns:
        mpz: Número primo grande
        
    Example:
        >>> primo = gerar_primo_grande(100)  # Primo de 100 dígitos
    """
    primos_gerados = carregar_primos_gerados(arquivo_registro)
    rs = random_state(int(time.time()))
    
    while True:
        # Gera um número aleatório com o número de dígitos especificado
        numero = gmpy2.mpz_random(rs, mpz(10**digitos - 10**(digitos-1))) + mpz(10**(digitos-1))
        # Garante que o número seja ímpar
        if numero % 2 == 0:
            numero += 1
        # Verifica se o número já foi gerado antes
        if numero in primos_gerados:
            continue
        # Testa se o número é primo
        if gmpy2.is_prime(numero):
            salvar_primo_gerado(numero, arquivo_registro)
            return numero


