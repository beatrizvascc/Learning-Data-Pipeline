"""
Módulo de Números Primos

Funções para geração, verificação e análise de números primos.
"""

from .gerador_primos import gerar_primos
from .primalidade import miller_rabin, eh_primo

__all__ = ['gerar_primos', 'miller_rabin', 'eh_primo']


