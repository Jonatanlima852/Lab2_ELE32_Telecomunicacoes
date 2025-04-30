import numpy as np
from typing import Tuple, List
from Lab2.gerar_grafo_ldpc import converter_H_para_AB

def decodificador_bit_flipping(palavra_recebida: np.ndarray, 
                             A: np.ndarray, 
                             B: np.ndarray, 
                             max_iteracoes: int = 100) -> Tuple[np.ndarray, int]:
    """
    Implementa o algoritmo de decodificação bit-flipping para códigos LDPC.
    
    Args:
        palavra_recebida: Palavra código recebida (array de N bits)
        A: Matriz N x dv de conexões dos v-nodes
        B: Matriz M x dc de conexões dos c-nodes
        max_iteracoes: Número máximo de iterações permitidas
        
    Returns:
        Tuple contendo:
        - Palavra decodificada
        - Número de iterações realizadas
    """
    N = len(palavra_recebida)
    M = B.shape[0]
    
    # Copia a palavra recebida para não modificar o original
    palavra_atual = np.copy(palavra_recebida)
    
    for iteracao in range(max_iteracoes):
        # Passo 1: Calcular as equações de paridade
        sindrome = np.zeros(M, dtype=int)
        for i in range(M):
            # XOR dos bits conectados ao c-node i
            sindrome[i] = np.sum(palavra_atual[B[i, :]]) % 2
        
        # Se todas as equações de paridade são satisfeitas, termina
        if np.all(sindrome == 0):
            return palavra_atual, iteracao + 1
        
        # Passo 2: Para cada bit, conta o número de equações de paridade insatisfeitas
        num_equacoes_insatisfeitas = np.zeros(N, dtype=int)
        for i in range(N):
            # Para cada c-node conectado ao v-node i
            for c_node in A[i, :]:
                if sindrome[c_node] == 1:
                    num_equacoes_insatisfeitas[i] += 1
        
        # Passo 3: Inverte os bits que participam do maior número de equações insatisfeitas
        max_insatisfeitas = np.max(num_equacoes_insatisfeitas)
        if max_insatisfeitas == 0:
            break
        
        # Inverte todos os bits que participam do número máximo de equações insatisfeitas
        bits_para_inverter = num_equacoes_insatisfeitas == max_insatisfeitas
        palavra_atual[bits_para_inverter] = 1 - palavra_atual[bits_para_inverter]
    
    return palavra_atual, max_iteracoes

# Função auxiliar para teste
def testar_decodificador():
    """
    Função para testar o decodificador com um exemplo simples
    """
    # Carrega uma das matrizes geradas anteriormente
    H = np.load("matriz_ldpc_N100.npy")  # Ajuste o nome do arquivo conforme necessário
    
    # Converte H para matrizes A e B
    A, B = converter_H_para_AB(H)
    
    # Cria uma palavra código de teste (todos zeros, por exemplo)
    N = H.shape[1]
    palavra_original = np.zeros(N, dtype=int)
    
    # Adiciona alguns erros aleatórios
    p_erro = 0.1  # probabilidade de erro
    erros = np.random.random(N) < p_erro
    palavra_recebida = (palavra_original + erros) % 2
    
    # Decodifica
    palavra_decodificada, num_iteracoes = decodificador_bit_flipping(
        palavra_recebida, A, B
    )
    
    # Exibe resultados
    print(f"Número de erros introduzidos: {np.sum(erros)}")
    print(f"Número de erros após decodificação: {np.sum(palavra_decodificada != palavra_original)}")
    print(f"Número de iterações: {num_iteracoes}")

if __name__ == "__main__":
    testar_decodificador()
