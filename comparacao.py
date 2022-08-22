import numpy as np
import matplotlib.pyplot as plt
import timeit
import QR

seed = 5

tc = []
tm = []
th = []
rng = np.random.default_rng(seed)
for i in range(400, 501):
    x = np.random.uniform(size = (500, i))
    tc.append(timeit.timeit("QR.gsc(x)", setup = "from __main__ import x, QR.gsc", number = 1))
    tm.append(timeit.timeit("QR.gsm(x)", setup = "from __main__ import x, QR.gsm", number = 1))
    th.append(timeit.timeit("QR.hh(x)", setup = "from __main__ import x, QR.hh", number = 1))

n = range(400, 501)
fig = plt.figure(figsize = (24, 16))
grafico = fig.add_subplot(111)
grafico.plot(n, tc, color = "blue")
grafico.plot(n, tm, color = "red")
grafico.plot(n, th, color = "purple")
plt.xlabel("Número de colunas (n)", fontsize = 16)
plt.ylabel("Tempo de execução (em segundos)", fontsize = 16)
plt.grid()
fig.savefig("comparacao-tempo.png")