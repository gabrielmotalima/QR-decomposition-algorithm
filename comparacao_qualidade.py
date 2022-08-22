import magic_square
import scipy.linalg as LA
import numpy as np
import QR

#para gsc
with open("GSC-tabelas.txt", "w") as arq:
    for j in range(10, 21):
        M = magic_square.magic(j)
        H = LA.hilbert(j)
        QM, RM = QR.gsc(M)
        QH, RH = QR.gsc(H)
        arq.write(f"{j} & {LA.norm(QM@RM - M, ord = np.inf)} & {LA.norm(QM.T @ QM - np.identity(j), ord = np.inf)} & {LA.norm(QH@RH - H, ord = np.inf)} & {LA.norm(QH.T @ QH - np.identity(j), ord = np.inf)} \\\\\n ")
        
#para gsm
with open("GSM-tabelas.txt", "w") as arq:
    for j in range(10, 21):
        M = magic_square.magic(j)
        H = LA.hilbert(j)
        QM, RM = QR.gsm(M)
        QH, RH = QR.gsm(H)
        arq.write(f"{j} & {LA.norm(QM@RM - M, ord = np.inf)} & {LA.norm(QM.T @ QM - np.identity(j), ord = np.inf)} & {LA.norm(QH@RH - H, ord = np.inf)} & {LA.norm(QH.T @ QH - np.identity(j), ord = np.inf)} \\\\\n ")

# para hh
with open("HH-tabelas.txt", "w") as arq:
    for j in range(10, 21):
        M = magic_square.magic(j)
        H = LA.hilbert(j)
        QM, RM = QR.hh(M)
        QH, RH = QR.hh(H)
        arq.write(f"{j} & {LA.norm(QM@RM - M, ord = np.inf)} & {LA.norm(QM.T @ QM - np.identity(j), ord = np.inf)} & {LA.norm(QH@RH - H, ord = np.inf)} & {LA.norm(QH.T @ QH - np.identity(j), ord = np.inf)} \\\\\n ")