import numpy as np
from ldpc import criar_matriz_verificacao_ldpc, calcular_taxa_codigo

def encontrar_N_proximo(valor_alvo: int, dv: int, dc: int) -> int:
    """
    Encontra o valor de N mais próximo do valor alvo que satisfaz as restrições do código LDPC.
    
    Args:
        valor_alvo: Valor desejado para N
        dv: Grau dos nós variáveis
        dc: Grau dos nós de verificação
        
    Returns:
        Valor válido de N mais próximo do alvo
    """
    # N*dv deve ser divisível por dc para termos um número inteiro de linhas
    # Encontra o múltiplo mais próximo
    N = valor_alvo
    while (N * dv) % dc != 0:
        N += 1
    return N

def gerar_matrizes_ldpc():
    """
    Gera matrizes LDPC com taxa aproximada de 4/7 para diferentes valores de N.
    Usa dv=6 e dc=14 para atingir a taxa desejada.
    """
    # Parâmetros fixos
    dv = 6  # grau dos nós variáveis
    dc = 14  # grau dos nós de verificação
    
    # Taxa teórica = 1 - dv/dc = 1 - 6/14 ≈ 0.571 (próximo de 4/7 ≈ 0.571)
    taxa_teorica = 1 - dv/dc
    
    # Valores alvo para N
    valores_alvo = [100, 200, 500, 1000]
    
    # Para cada valor alvo, encontra o N mais próximo válido e gera a matriz
    for valor_alvo in valores_alvo:
        N = encontrar_N_proximo(valor_alvo, dv, dc)
        
        print(f"\nGerando matriz para N ≈ {valor_alvo}")
        print(f"N ajustado = {N}")
        
        try:
            H = criar_matriz_verificacao_ldpc(N, dv, dc)
            
            # Calcula e exibe as propriedades da matriz
            M = H.shape[0]
            taxa_real = calcular_taxa_codigo(H)
            
            print(f"Dimensões da matriz H: {M} x {N}")
            print(f"Taxa teórica: {taxa_teorica:.6f}")
            print(f"Taxa real: {taxa_real:.6f}")
            
            # Salva a matriz em um arquivo
            nome_arquivo = f"matriz_ldpc_N{N}.npy"
            np.save(nome_arquivo, H)
            print(f"Matriz salva em: {nome_arquivo}")
            
            # Verificações adicionais
            print("\nVerificações:")
            print(f"- Todos v-nodes têm grau {dv}:", all(np.sum(H[:, j]) == dv for j in range(N)))
            print(f"- Todos c-nodes têm grau {dc}:", all(np.sum(H[i, :]) == dc for i in range(M)))
            
        except ValueError as e:
            print(f"Erro ao gerar matriz: {e}")

if __name__ == "__main__":
    print("Iniciando geração das matrizes LDPC...")
    print("Parâmetros: dv=6, dc=14 (taxa alvo ≈ 4/7)")
    gerar_matrizes_ldpc()
