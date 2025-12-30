"""
Análise de Fourier (FFT)

Transformada de Fourier para análise espectral de dados.
"""

import numpy as np
import matplotlib.pyplot as plt


def analise_fourier(dados, salvar_grafico=None, titulo="Transformada de Fourier"):
    """
    Realiza análise de Fourier (FFT) dos dados.
    
    Args:
        dados (array-like): Dados para análise
        salvar_grafico (str, optional): Caminho para salvar o gráfico
        titulo (str): Título do gráfico
        
    Returns:
        tuple: (frequencias, magnitudes)
        
    Example:
        >>> gaps = [2, 4, 2, 4, 6, 2, 6, 4]
        >>> freq, mag = analise_fourier(gaps)
    """
    dados_array = np.array(dados)
    fft = np.fft.fft(dados_array)
    freq = np.fft.fftfreq(len(dados_array))
    
    magnitudes = np.abs(fft)
    
    if salvar_grafico:
        plt.figure(figsize=(10, 5))
        plt.plot(freq[:len(freq)//2], magnitudes[:len(magnitudes)//2])
        plt.title(titulo)
        plt.xlabel('Frequência')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.savefig(salvar_grafico)
        plt.close()
    
    return freq, magnitudes


