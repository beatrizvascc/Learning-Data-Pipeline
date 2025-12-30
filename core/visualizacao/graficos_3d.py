"""
Visualizações 3D

Funções para criar visualizações 3D de dados (espirais, toroides, etc).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualizar_espiral(dados, salvar_grafico=None, titulo="Espiral Logarítmica"):
    """
    Visualiza dados em uma espiral logarítmica.
    
    Args:
        dados (array-like): Dados para visualizar
        salvar_grafico (str, optional): Caminho para salvar o gráfico
        titulo (str): Título do gráfico
        
    Example:
        >>> primos = [2, 3, 5, 7, 11, 13, 17, 19]
        >>> visualizar_espiral(primos)
    """
    dados_array = np.array(dados)
    theta = dados_array * 0.5
    r = np.sqrt(dados_array)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=theta, cmap='hsv', s=10, alpha=0.75)
    plt.axis('off')
    plt.title(titulo)
    
    if salvar_grafico:
        plt.savefig(salvar_grafico)
        plt.close()
    else:
        plt.show()


def visualizar_toroide(dados, salvar_grafico=None, titulo="Toroide"):
    """
    Visualiza dados em um toroide 3D.
    
    Args:
        dados (array-like): Dados para visualizar
        salvar_grafico (str, optional): Caminho para salvar o gráfico
        titulo (str): Título do gráfico
        
    Example:
        >>> primos = [2, 3, 5, 7, 11, 13, 17, 19]
        >>> visualizar_toroide(primos)
    """
    dados_array = np.array(dados)
    R = 10
    r_small = 2
    u = dados_array * 0.1
    v = dados_array * 0.07

    x = (R + r_small * np.cos(v)) * np.cos(u)
    y = (R + r_small * np.cos(v)) * np.sin(u)
    z = r_small * np.sin(v)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=dados_array, cmap='plasma', s=5)
    ax.set_title(titulo)
    
    if salvar_grafico:
        plt.savefig(salvar_grafico)
        plt.close()
    else:
        plt.show()


