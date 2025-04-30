import numpy as np
import matplotlib.pyplot as plt
from Lab2.gerar_grafo_ldpc import criar_matriz_verificacao_ldpc, converter_H_para_AB
from Lab2.decodificador_ldpc import decodificador_bit_flipping
from Lab1.main import CodificadorHamming, DecodificadorHamming


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


def simular_transmissao_ldpc(H, erro_canal, num_palavras_base, num_palavras_baixo_erro, limite_erro):
    """
    Simula transmissão usando código LDPC.
    
    Args:
        H: Matriz de verificação de paridade
        erro_canal: Lista de probabilidades de erro do canal
        num_palavras_base: Número de palavras para probabilidades de erro >= limite_erro
        num_palavras_baixo_erro: Número de palavras para probabilidades de erro < limite_erro
        limite_erro: Limite de probabilidade de erro para usar num_palavras_baixo_erro
    """
    print(f"Iniciando simulação LDPC com {num_palavras_base} palavras para p >= {limite_erro} e {num_palavras_baixo_erro} palavras para p < {limite_erro}...")
    
    M, N = H.shape
    k = N - M  # dimensão da palavra de informação
    
    # Converter H para matrizes A e B para o decodificador
    print("Convertendo matriz H para matrizes A e B...")
    A, B = converter_H_para_AB(H)
    print(f"Matrizes A e B geradas. Dimensões A: {A.shape}, B: {B.shape}")
    
    resultados = []
    for p in erro_canal:
        num_palavras = num_palavras_baixo_erro if p < limite_erro else num_palavras_base
        print(f"\nSimulando transmissão LDPC com p = {p} usando {num_palavras} palavras")
        bits_errados = 0
        total_erros_inseridos = 0
        
        for i in range(num_palavras):
            # Como a palavra de informação é tudo 0, a palavra codificada também é tudo 0
            v = np.zeros(N, dtype=int)
            
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


def simular_transmissao_hamming_com_matriz_ldpc(H, erro_canal, num_palavras):
    """
    Simula transmissão usando código Hamming com a mesma matriz H do LDPC.
    
    Args:
        H: Matriz de verificação de paridade (a mesma usada no LDPC)
        erro_canal: Lista de probabilidades de erro do canal
        num_palavras: Número de palavras para simulação
    """
    print(f"Iniciando simulação Hamming usando matriz LDPC com {num_palavras} palavras...")
    
    M, N = H.shape
    k = N - M  # dimensão da palavra de informação
    
    # Transpor H para formato esperado pelo DecodificadorHamming
    H_T = H.T
    
    # Criar uma matriz G básica para o decodificador Hamming
    # G = [I_k | P] onde I_k é a matriz identidade k×k

    G = np.zeros((k, N), dtype=int)
    G[:, :k] = np.eye(k, dtype=int)  # Parte identidade
    
    # Inicializar apenas o decodificador
    decodificador = DecodificadorHamming(H_T, G)
    
    resultados = []
    for p in erro_canal:
        print(f"\nSimulando transmissão Hamming (com matriz LDPC) com p = {p}")
        bits_errados = 0
        total_erros_inseridos = 0
        
        for i in range(num_palavras):
            # Como a palavra de informação é tudo 0, a palavra codificada também é tudo 0
            v = np.zeros(N, dtype=int)
            
            # Transmite pelo canal BSC
            canal = CanalBSC(p)
            r, erros_inseridos = canal.transmitir(v)
            total_erros_inseridos += erros_inseridos
            
            # Calcular a síndrome
            sindrome = np.dot(r, H_T) % 2
            
            # Simplificação: para palavra de tudo 0, qualquer bit 1 em u_decodificado é erro
            if np.sum(sindrome) == 0:
                # Sem erros detectados
                u_decodificado = r[:k]
            else:
                # Com erros detectados, usa o decodificador Hamming
                u_decodificado, _, _ = decodificador.decodificar(r)
            
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


def plotar_comparacao_codigos(resultados_ldpc, resultados_hamming, sem_codigo, nome_ldpc, nome_hamming):
    """
    Plota os resultados da simulação para comparação entre códigos LDPC e Hamming.
    """
    print("\nGerando gráfico comparativo...")
    plt.figure(figsize=(12, 8))
    
    # Extrai valores para cada conjunto de resultados
    p_ldpc, prob_ldpc = zip(*resultados_ldpc)
    p_hamming, prob_hamming = zip(*resultados_hamming)
    p_sem, prob_sem = zip(*sem_codigo)
    
    # Plota os gráficos
    plt.loglog(p_ldpc, prob_ldpc, 'o-', label=nome_ldpc, color='blue')
    plt.loglog(p_hamming, prob_hamming, 's-', label=nome_hamming, color='red')
    plt.loglog(p_sem, prob_sem, '--', label='Sem codificação', color='black')
    
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls="-")
    plt.xlabel('Probabilidade de erro do canal (p)')
    plt.ylabel('Probabilidade de erro de bit')
    plt.title(f'Comparação das taxas de erro de bit - {nome_ldpc} vs {nome_hamming}')
    plt.legend()
    plt.savefig(f'comparacao_{nome_ldpc.replace(" ", "_").lower()}_{nome_hamming.replace(" ", "_").lower()}.png')
    print(f"Gráfico salvo como 'comparacao_{nome_ldpc.replace(' ', '_').lower()}_{nome_hamming.replace(' ', '_').lower()}.png'")
    plt.show()


def plotar_comparacao_ldpc(resultados_100, resultados_200, resultados_500, resultados_1000, resultados_hamming, sem_codigo):
    """
    Plota os resultados da simulação LDPC.
    """
    print("\nGerando gráfico comparativo de todos os códigos...")
    plt.figure(figsize=(12, 8))
    
    # Extrai valores para cada conjunto de resultados
    p_100, prob_100 = zip(*resultados_100)
    p_200, prob_200 = zip(*resultados_200)
    p_500, prob_500 = zip(*resultados_500)
    p_1000, prob_1000 = zip(*resultados_1000)
    p_hamming, prob_hamming = zip(*resultados_hamming)
    p_sem, prob_sem = zip(*sem_codigo)
    
    # Plota os gráficos
    plt.loglog(p_100, prob_100, 'o-', label='LDPC N≈100', color='blue')
    plt.loglog(p_200, prob_200, 's-', label='LDPC N≈200', color='red')
    plt.loglog(p_500, prob_500, '^-', label='LDPC N≈500', color='green')
    plt.loglog(p_1000, prob_1000, '*-', label='LDPC N≈1000', color='purple')
    plt.loglog(p_hamming, prob_hamming, 'D-', label='Hamming (equivalente)', color='orange')
    plt.loglog(p_sem, prob_sem, '--', label='Sem codificação', color='black')
    
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", ls="-")
    plt.xlabel('Probabilidade de erro do canal (p)')
    plt.ylabel('Probabilidade de erro de bit')
    plt.title('Comparação das taxas de erro de bit - Códigos LDPC vs Hamming')
    plt.legend()
    plt.savefig('comparacao_todos_codigos.png')
    print("Gráfico salvo como 'comparacao_todos_codigos.png'")
    plt.show()


def simular_sem_codigo(erro_canal, num_bits):
    """
    Simula transmissão sem codificação (y=x).
    """
    print("Calculando caso sem codificação...")
    return [(p, p) for p in erro_canal]


def main():
    print("=== Iniciando simulação de códigos LDPC e Hamming ===")
    
    # Parâmetros do código LDPC
    dv = 6  # grau dos nós variáveis
    dc = 14  # grau dos nós de verificação
    print(f"Parâmetros LDPC: dv={dv}, dc={dc}, taxa = {1-dv/dc:.4f}")
    
    # Valores de N
    N_valores = [100, 200, 500, 1000]
    
    # Probabilidades de erro do canal
    erro_canal = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
    print(f"Probabilidades de erro: {erro_canal}")
    
    # Número de palavras para simulação
    num_palavras_base = 500  # para p >= limite_erro
    num_palavras_baixo_erro = 5000  # para p < limite_erro
    num_palavras_hamming = 20000  # maior número para Hamming
    limite_erro = 0.04  # limite para usar mais palavras
    print(f"Número de palavras para p >= {limite_erro}: {num_palavras_base}")
    print(f"Número de palavras para p < {limite_erro}: {num_palavras_baixo_erro}")
    print(f"Número de palavras para Hamming: {num_palavras_hamming}")
    
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
        
        # Simula transmissão (sem precisar de G)
        resultados = simular_transmissao_ldpc(H, erro_canal, num_palavras_base, num_palavras_baixo_erro, limite_erro)
        resultados_todos.append(resultados)
    
    # Cria uma matriz H menor para Hamming, mantendo a mesma taxa (1-dv/dc)
    N_hamming = 20  # Tamanho menor, mas mantendo a mesma taxa
    while (N_hamming * dv) % dc != 0:
        N_hamming += 1
    
    print(f"\n=== Criando matriz menor para Hamming: N = {N_hamming} ===")
    H_hamming = criar_matriz_verificacao_ldpc(N_hamming, dv, dc)
    print(f"Matriz H para Hamming criada. Dimensões: {H_hamming.shape}")
    
    # Simula transmissão Hamming usando a matriz LDPC menor
    resultados_hamming = simular_transmissao_hamming_com_matriz_ldpc(H_hamming, erro_canal, num_palavras_hamming)
    
    # Simula caso sem código
    num_bits = 100000  # para o caso sem código
    resultados_sem_codigo = simular_sem_codigo(erro_canal, num_bits)
    
    # Plota comparação de todos os códigos
    plotar_comparacao_ldpc(
        resultados_todos[0],  # N≈100
        resultados_todos[1],  # N≈200
        resultados_todos[2],  # N≈500
        resultados_todos[3],  # N≈1000
        resultados_hamming,   # Hamming (N menor)
        resultados_sem_codigo
    )
    
    print("\n=== Simulação concluída ===")


if __name__ == "__main__":
    main()