import numpy as np
from typing import Tuple

def criar_matriz_verificacao_ldpc(N: int, dv: int, dc: int) -> np.ndarray:
    """
    Cria uma matriz de verificação de paridade H para um código LDPC regular.
    
    Args:
        N: Número de colunas (comprimento da palavra código)
        dv: Grau dos nós variáveis (número de 1s por coluna)
        dc: Grau dos nós de verificação (número de 1s por linha)
        
    Returns:
        H: Matriz de verificação de paridade (M x N)
    """
    # Calcula o número de linhas M baseado nos parâmetros
    # N*dv = M*dc (número total de 1s deve ser igual olhando por linhas ou colunas)
    M = (N * dv) // dc
    
    if (N * dv) % dc != 0:
        raise ValueError(f"Parâmetros inválidos: N*dv ({N*dv}) deve ser divisível por dc ({dc})")
    
    # Inicializa matriz com zeros
    H = np.zeros((M, N), dtype=int)
    
    # Para cada coluna (v-node)
    for j in range(N):
        # Lista de linhas disponíveis (onde ainda podemos colocar 1s)
        linhas_disponiveis = [i for i in range(M) if np.sum(H[i]) < dc]
        
        # Se não há linhas suficientes disponíveis, recomeça
        if len(linhas_disponiveis) < dv:
            return criar_matriz_verificacao_ldpc(N, dv, dc)
        
        # Escolhe aleatoriamente dv linhas para colocar 1s
        linhas_escolhidas = np.random.choice(linhas_disponiveis, size=dv, replace=False)
        
        for i in linhas_escolhidas:
            H[i, j] = 1
    
    # Verifica se todas as restrições foram atendidas
    if not _verificar_matriz(H, dv, dc):
        # Se não foram, tenta criar novamente
        return criar_matriz_verificacao_ldpc(N, dv, dc)
    
    return H

def _verificar_matriz(H: np.ndarray, dv: int, dc: int) -> bool:
    """
    Verifica se a matriz H atende a todas as restrições do código LDPC regular.
    """
    M, N = H.shape
    
    # Verifica grau dos v-nodes (colunas)
    if not all(np.sum(H[:, j]) == dv for j in range(N)):
        return False
    
    # Verifica grau dos c-nodes (linhas)
    if not all(np.sum(H[i, :]) == dc for i in range(M)):
        return False
    
    return True

def calcular_taxa_codigo(H: np.ndarray) -> float:
    """
    Calcula a taxa do código LDPC.
    
    Args:
        H: Matriz de verificação de paridade
        
    Returns:
        Taxa do código (R = k/n)
    """
    M, N = H.shape
    return (N - M) / N

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo com parâmetros pequenos para teste
    N = 20  # comprimento da palavra código
    dv = 3  # grau dos nós variáveis
    dc = 4  # grau dos nós de verificação
    
    try:
        H = criar_matriz_verificacao_ldpc(N, dv, dc)
        print("Matriz de verificação de paridade H:")
        print(H)
        print(f"\nDimensões da matriz: {H.shape}")
        print(f"Taxa do código: {calcular_taxa_codigo(H):.3f}")
        
        # Verifica as propriedades
        print("\nVerificando propriedades:")
        print(f"Todos v-nodes têm grau {dv}:", all(np.sum(H[:, j]) == dv for j in range(N)))
        print(f"Todos c-nodes têm grau {dc}:", all(np.sum(H[i, :]) == dc for i in range(H.shape[0])))
        
    except ValueError as e:
        print(f"Erro: {e}")
