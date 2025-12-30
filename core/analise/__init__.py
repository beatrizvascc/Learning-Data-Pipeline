"""
Módulo de Análise Matemática

Funções para análise de dados: Fourier, Wavelet, PCA, estatísticas.
"""

from .fourier import analise_fourier
from .wavelet import analise_wavelet
from .pca import analise_pca

__all__ = ['analise_fourier', 'analise_wavelet', 'analise_pca']


