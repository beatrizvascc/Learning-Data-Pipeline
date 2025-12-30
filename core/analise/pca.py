"""
Análise de Componentes Principais (PCA)

Redução de dimensionalidade usando PCA.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def analise_pca(dados, n_components=2, salvar_grafico=None):
    """
    Realiza análise de Componentes Principais (PCA).
    
    Args:
        dados (array-like): Dados para análise (shape: n_samples, n_features)
        n_components (int): Número de componentes principais
        salvar_grafico (str, optional): Caminho para salvar o gráfico (apenas para 2D)
        
    Returns:
        tuple: (componentes, pca_object)
            - componentes: Dados transformados
            - pca_object: Objeto PCA treinado
            
    Example:
        >>> dados = np.random.rand(100, 5)
        >>> comp, pca = analise_pca(dados, n_components=2)
    """
    dados_array = np.array(dados)
    
    if dados_array.ndim == 1:
        dados_array = dados_array.reshape(-1, 1)
    
    pca = PCA(n_components=n_components)
    componentes = pca.fit_transform(dados_array)
    
    if salvar_grafico and n_components == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(componentes[:, 0], componentes[:, 1], s=5, alpha=0.6)
        plt.title('Análise de Componentes Principais (PCA)')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.grid(True)
        plt.savefig(salvar_grafico)
        plt.close()
    
    return componentes, pca


