import numpy as np
import scipy.linalg as LA
import QR


#testes para GSC com matrizes bem condicionadas
seed = 7
rng = np.random.default_rng(seed)
correto = []
for i in range(30):
    H = LA.hadamard(2**3)/np.sqrt(8)
    D = np.zeros((8,8))
    np.fill_diagonal(D, rng.integers(1, 31, size = (8)))
    X = H@D
    Q, R = QR.gsc(X)
    if LA.norm(Q - H, ord = np.inf) < 10**(-15) and LA.norm(R - D, ord = np.inf) < 10**(-14):
        correto.append(True)
    else:
        correto.append(False)
        break
print("A função gsc produziu resultados corretos para matrizes de Hadamard?", all(correto))





#testes para GSM com matrizes bem condicionadas
correto = []
for i in range(30):
    H = LA.hadamard(2**3)/np.sqrt(8)
    D = np.zeros((8,8))
    np.fill_diagonal(D, rng.integers(1, 31, size = (8)))
    
    X = H@D
    Q, R = QR.gsm(X)
    if LA.norm(Q - H, ord = np.inf) < 10**(-15) and LA.norm(R - D, ord = np.inf) < 10**(-14):
        correto.append(True)
    else:
        correto.append(False)
        break
print("\nA função gsm produziu resultados corretos para matrizes de Hadamard?", all(correto))

#testes para GSM com matrizes aleatórias
correto = []
for i in range(50, 81):
    X = np.random.uniform(-1, 1, size = (80, i))
    Q, R = QR.gsm(X)
    if LA.norm(Q.T@Q - np.identity(i), ord = np.inf) < 10**(-12) and LA.norm(Q@R -X, ord = np.inf) < 10**(-12):
        correto.append(True)
    else:
        correto.append(False)
        break
print("A função gsm produziu resultados corretos para matrizes aleatórias?", all(correto))


#testes para HH com matrizes aleatórias
correto = []
for i in range(50, 81):
    X = np.random.uniform(-1, 1, size = (80, i))
    Q, R = QR.hh(X)
    if LA.norm(Q.T@Q - np.identity(80), ord = np.inf) < 10**(-12) and LA.norm(Q@R -X, ord = np.inf) < 10**(-12):
        correto.append(True)
    else:
        correto.append(False)
        break
print("\nA função gsm produziu resultados corretos para matrizes aleatórias?", all(correto))