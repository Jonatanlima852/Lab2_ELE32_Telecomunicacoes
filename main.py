import numpy as np
import matplotlib.pyplot as plt
from matriz_verificacao import criar_matriz_verificacao_ldpc
from decodificador_ldpc import converter_H_para_AB, decodificador_bit_flipping


class CanalBSC:
    """
    Simula um Canal Binário Simétrico (BSC) com probabilidade p de erro.
    """
    def __init__(self, p):
        """
        Inicializa o canal com probabilidade p de erro.
        
        Args:
            p: Probabilidade de inverter um bit
        """
        self.p = p
    
    def transmitir(self, palavra):
        """
        Transmite uma palavra através do canal, com possibilidade de erro.
        
        Args:
            palavra: Array ou lista de bits a ser transmitida
            
        Returns:
            Array com a palavra possivelmente modificada
        """
        resultado = np.copy(palavra)
        erros = np.random.random(len(resultado)) < self.p
        resultado[erros] = 1 - resultado[erros]  # Inverte os bits com erro
        return resultado, np.sum(erros)


def obter_matriz_G(H: np.ndarray) -> np.ndarray:
    """
    Obtém a matriz geradora G a partir da matriz de verificação de paridade H.
    G = [I | P], onde P = (H_1^T * H_2^(-1))^T
    
    Args:
        H: Matriz de verificação de paridade
    
    Returns:
        G: Matriz geradora
    """
    print("Calculando matriz geradora G a partir de H...")
    M, N = H.shape
    k = N - M  # dimensão da palavra de informação
    
    # Reorganiza H para forma [H_1 | H_2] onde H_2 é M x M
    # Encontra M colunas linearmente independentes para H_2
    H_copy = H.copy()
    selected_cols = []
    remaining_cols = list(range(N))
    rank = 0
    
    # Encontra M colunas linearmente independentes
    while rank < M and remaining_cols:
        for col in remaining_cols[:]:
            temp_cols = selected_cols + [col]
            temp_matrix = H_copy[:, temp_cols]
            if np.linalg.matrix_rank(temp_matrix) > rank:
                selected_cols.append(col)
                rank = np.linalg.matrix_rank(H_copy[:, selected_cols])
                remaining_cols.remove(col)
                break
    
    if rank < M:
        raise ValueError("Matriz H não tem posto completo")
    
    # Reorganiza H para [H_1 | H_2]
    perm = remaining_cols + selected_cols
    H_perm = H_copy[:, perm]
    H_1 = H_perm[:, :k]
    H_2 = H_perm[:, k:]
    
    try:
        # Calcula P = (H_1^T * H_2^(-1))^T
        H_2_inv = np.linalg.inv(H_2)
        P = np.dot(H_1.T, H_2_inv).T % 2
        
        # Constrói G = [I | -P]
        I_k = np.eye(k, dtype=int)
        G = np.zeros((k, N), dtype=int)
        G[:, :k] = I_k
        G[:, k:] = P.T  # Transpõe P para ter as dimensões corretas
        
        # Desfaz a permutação
        G = G[:, np.argsort(perm)]
        
        print(f"Matriz G calculada com sucesso. Dimensões: {G.shape}")
        return G
    except np.linalg.LinAlgError:
        raise ValueError("Matriz H_2 não é inversível")


def simular_transmissao_ldpc(H, G, erro_canal, num_palavras):
    """
    Simula transmissão usando código LDPC.
    """
    print(f"Iniciando simulação LDPC com {num_palavras} palavras por probabilidade...")
    
    k = G.shape[0]  # dimensão da palavra de informação
    N = H.shape[1]  # comprimento da palavra código
    
    # Converter H para matrizes A e B para o decodificador
    print("Convertendo matriz H para matrizes A e B...")
    A, B = converter_H_para_AB(H)
    print(f"Matrizes A e B geradas. Dimensões A: {A.shape}, B: {B.shape}")
    
    resultados = []
    for p in erro_canal:
        print(f"\nSimulando transmissão LDPC com p = {p}")
        bits_errados = 0
        total_erros_inseridos = 0
        
        for i in range(num_palavras):
            # Palavra de informação com todos os bits 0
            u = np.zeros(k, dtype=int)
            
            # Codifica
            v = np.dot(u, G) % 2
            
            # Transmite pelo canal BSC
            canal = CanalBSC(p)
            r, erros_inseridos = canal.transmitir(v)
            total_erros_inseridos += erros_inseridos
            
            # Decodifica
            v_decodificado, iteracoes = decodificador_bit_flipping(r, A, B)
            u_decodificado = v_decodificado[:k]  # para códigos sistemáticos
            
            # Conta erros (como u é tudo 0, qualquer 1 em u_decodificado é erro)
            erros_decodificacao = np.sum(u_decodificado)
            bits_errados += erros_decodificacao
            
            # Log a cada 10% de progresso
            if (i+1) % (num_palavras // 10) == 0 or (i+1) == num_palavras:
                progresso = (i+1) * 100 // num_palavras
                print(f"  Progresso: {progresso}% ({i+1}/{num_palavras})")
        
        prob_erro = bits_errados / (num_palavras * k)
        resultados.append((p, prob_erro))
        print(f"  Erros inseridos: {total_erros_inseridos} bits (média: {total_erros_inseridos/num_palavras:.2f} por palavra)")
        print(f"  Erros após decodificação: {bits_errados} bits (média: {bits_errados/num_palavras:.2f} por palavra)")
        print(f"  Probabilidade de erro para p = {p}: {prob_erro:.8f}")
    
    return resultados


def plotar_comparacao_ldpc(resultados_100, resultados_200, resultados_500, sem_codigo):
    """
    Plota os resultados da simulação LDPC.
    """
    print("\nGerando gráfico comparativo...")
    plt.figure(figsize=(12, 8))
    
    # Extrai valores para cada conjunto de resultados
    p_100, prob_100 = zip(*resultados_100)
    p_200, prob_200 = zip(*resultados_200)
    p_500, prob_500 = zip(*resultados_500)
    p_sem, prob_sem = zip(*sem_codigo)
    
    # Plota os gráficos
    plt.loglog(p_100, prob_100, 'o-', label='LDPC N≈100', color='blue')
    plt.loglog(p_200, prob_200, 's-', label='LDPC N≈200', color='red')
    plt.loglog(p_500, prob_500, '^-', label='LDPC N≈500', color='green')
    plt.loglog(p_sem, prob_sem, '--', label='Sem codificação', color='black')
    
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls="-")
    plt.xlabel('Probabilidade de erro do canal (p)')
    plt.ylabel('Probabilidade de erro de bit')
    plt.title('Comparação das taxas de erro de bit - Códigos LDPC')
    plt.legend()
    plt.savefig('comparacao_ldpc.png')
    print("Gráfico salvo como 'comparacao_ldpc.png'")
    plt.show()


def simular_sem_codigo(erro_canal, num_bits):
    """
    Simula transmissão sem codificação (y=x).
    """
    print("Calculando caso sem codificação...")
    return [(p, p) for p in erro_canal]


def main():
    print("=== Iniciando simulação de códigos LDPC ===")
    
    # Parâmetros do código LDPC
    dv = 6  # grau dos nós variáveis
    dc = 14  # grau dos nós de verificação
    print(f"Parâmetros: dv={dv}, dc={dc}, taxa = {1-dv/dc:.4f}")
    
    # Valores de N
    N_valores = [100, 200, 500]
    
    # Probabilidades de erro do canal
    erro_canal = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    print(f"Probabilidades de erro: {erro_canal}")
    
    # Número de palavras para simulação
    num_palavras = 1000
    print(f"Número de palavras por probabilidade: {num_palavras}")
    
    resultados_todos = []
    
    # Simula para cada tamanho de código
    for N_alvo in N_valores:
        print(f"\n=== Simulando código LDPC com N ≈ {N_alvo} ===")
        
        # Cria matriz H
        N = N_alvo
        while (N * dv) % dc != 0:
            N += 1
        
        print(f"Valor de N ajustado: {N}")
        print(f"Criando matriz de verificação LDPC {N}x{(N*dv)//dc}...")
        H = criar_matriz_verificacao_ldpc(N, dv, dc)
        print(f"Matriz H criada. Dimensões: {H.shape}")
        
        G = obter_matriz_G(H)
        
        # Simula transmissão
        resultados = simular_transmissao_ldpc(H, G, erro_canal, num_palavras)
        resultados_todos.append(resultados)
    
    # Simula caso sem código
    num_bits = 1000000  # para o caso sem código
    resultados_sem_codigo = simular_sem_codigo(erro_canal, num_bits)
    
    # Plota comparação
    plotar_comparacao_ldpc(
        resultados_todos[0],  # N≈100
        resultados_todos[1],  # N≈200
        resultados_todos[2],  # N≈500
        resultados_sem_codigo
    )
    
    print("\n=== Simulação concluída ===")


if __name__ == "__main__":
    main()