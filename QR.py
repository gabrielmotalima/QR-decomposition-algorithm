import numpy as np

def gsc(X):
    """Utiliza o algoritmo clássico de Gram-Schmidt para realizar a fatoração QR de uma matriz X: m×n com posto(X) = n.
    Retorna as matrizes (numpy.ndarray) Q e R, com Q: m×n com colunas ortonormais entre si, R: n×n triangular superior e QR = X.
    Versão 1.2: 08 de Julho de 2022"""
    m, n = X.shape
    Q = np.array(X, dtype = "float64")
    R = np.zeros((n, n))
    
    R[0, 0] = np.linalg.norm(Q[:, 0])
    Q[:, 0] = Q[:, 0] / R[0,0]
    for j in range(1, n):
        #calcula as projeções da coluna atual sobre as anteriores e subtrai estas projeções dela
        R[:j-1, j] = Q[:, :j - 1].T @ Q[:, j] / np.diagonal(R[:j-1, :j-1])
        Q[:, j] = Q[:, j] - Q[:, :j-1] @ R[:j-1, j]
        #normalização
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / R[j, j]
        
    return Q, R





def gsm(X):
    """Utiliza o algoritmo modificado de Gram-Schmidt para realizar a fatoração QR da matriz X: m×n, com posto(x) = n.
    Retorna as matrizes (numpy.ndarray) Q e R tais que Q:m×n possui colunas ortonormais entre si, R:n×n é triangular superior e QR = X.
    Versão 1.1: 08 de Julho de 2022"""
    m, n = X.shape
    Q = np.array(X, dtype = "float64")
    R = np.zeros((n, n))
    for j in range(n):
        #normaliza a coluna atual
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / R[j, j]
        #calcula as projeções da coluna atual sobre as seguintes e subtrai delas essas projeções
        R[j, j+1:] = Q[:, j].T @ Q[:, j+1:]
        Q[:, j+1:] = Q[:, j+1:] - np.outer(Q[:, j], R[j, j+1:])
    return Q, R





def hh(X):
    """Utiliza o algoritmo de Householder para realizar a fatoração QR de uma matriz X: m×n.
    Retorna as matrizes (numpy.ndarray) Q  e R tais que Q:m×m é ortogonal, R:m×n é "triangular superior" e QR = X.
    Versão 1.3: 11 de Julho de 2022"""
    m, n = X.shape
    p = min((m, n))
    R = np.array(X, dtype = "float64")
    Q = np.identity(m)
    V = np.zeros((m, p))
    #V armazena os vetores v referentes às matrizes de Householder para que Q seja computada ao fim do processo de triangularização
    e1 = Q[:, 0]
    
    for j in range(p):
        #triangulariza X produzindo R
        x = R[j:, j]
        norma = np.linalg.norm(x)
        if norma != 0:
            #condição que evita divisão por zero
            v = np.sign(x[0]) * norma * e1[:m-j] + x
            v = v / np.linalg.norm(v)
            R[j:, j:] = R[j:, j:] - 2 * np.outer(v, (v @ R[j:, j:]))
            V[:m - j, j] = v
    
    for i in range(p-1, -1, -1):
        #constrói Q
        if np.linalg.norm(V[:m-i, i]) != 0:
            Q[i:, i:] = Q[i:, i:] - 2 * np.outer(V[:m- i, i], (V[:m-i, i] @ Q[i:, i:]))
    return Q, R

