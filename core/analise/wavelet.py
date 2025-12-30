"""
Análise Wavelet

Análise de sinais usando transformada Wavelet.
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt


def analise_wavelet(dados, wavelet='db1', level=6, salvar_grafico=None):
    """
    Realiza análise Wavelet dos dados.
    
    Args:
        dados (array-like): Dados para análise
        wavelet (str): Tipo de wavelet (padrão: 'db1' - Daubechies)
        level (int): Nível de decomposição
        salvar_grafico (str, optional): Caminho para salvar o gráfico
        
    Returns:
        list: Lista de coeficientes Wavelet por nível
        
    Example:
        >>> gaps = [2, 4, 2, 4, 6, 2, 6, 4]
        >>> coeffs = analise_wavelet(gaps, level=3)
    """
    dados_array = np.array(dados)
    coeficientes = pywt.wavedec(dados_array, wavelet, level=level)
    
    if salvar_grafico:
        plt.figure(figsize=(12, 8))
        for i, c in enumerate(coeficientes):
            plt.subplot(len(coeficientes), 1, i + 1)
            plt.plot(c)
            plt.title(f'Coeficiente Wavelet Nível {i + 1}')
        plt.tight_layout()
        plt.savefig(salvar_grafico)
        plt.close()
    
    return coeficientes


