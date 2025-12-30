"""
Geração de Chaves RSA

Funções para gerar chaves RSA usando números primos grandes.
"""

from Crypto.PublicKey import RSA


def gerar_chaves_rsa(bits=2048):
    """
    Gera um par de chaves RSA (pública e privada).
    
    Args:
        bits (int): Tamanho da chave em bits (1024, 2048 ou 4096)
        
    Returns:
        dict: Dicionário com chaves em diferentes formatos:
            - privada_pem: Chave privada em formato PEM
            - publica_pem: Chave pública em formato PEM
            - publica_ssh: Chave pública em formato OpenSSH
            
    Example:
        >>> chaves = gerar_chaves_rsa(2048)
        >>> print(chaves['publica_pem'])
    """
    if bits not in [1024, 2048, 4096]:
        raise ValueError("Tamanho da chave deve ser 1024, 2048 ou 4096 bits")
    
    chave = RSA.generate(bits)

    chave_privada_pem = chave.export_key()
    chave_publica_pem = chave.publickey().export_key()
    chave_publica_ssh = chave.publickey().export_key(format='OpenSSH')

    return {
        "privada_pem": chave_privada_pem.decode(),
        "publica_pem": chave_publica_pem.decode(),
        "publica_ssh": chave_publica_ssh.decode()
    }


def salvar_chaves(chaves, prefixo="id_rsa"):
    """
    Salva as chaves RSA em arquivos.
    
    Args:
        chaves (dict): Dicionário retornado por gerar_chaves_rsa()
        prefixo (str): Prefixo para os nomes dos arquivos
        
    Returns:
        list: Lista de arquivos criados
    """
    arquivos_criados = []
    
    # Chave privada
    with open(f"{prefixo}", 'w') as f:
        f.write(chaves["privada_pem"])
    arquivos_criados.append(f"{prefixo}")
    
    # Chave pública SSH
    with open(f"{prefixo}.pub", 'w') as f:
        f.write(chaves["publica_ssh"])
    arquivos_criados.append(f"{prefixo}.pub")
    
    # Chave pública PEM
    with open(f"{prefixo}_public.pem", 'w') as f:
        f.write(chaves["publica_pem"])
    arquivos_criados.append(f"{prefixo}_public.pem")
    
    return arquivos_criados


