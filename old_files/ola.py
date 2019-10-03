from numpy import diag, empty, ones, trace
from numpy.linalg import eigvalsh
from numpy.random import RandomState
from scipy.linalg import sqrtm

random = RandomState(0)

rhos = [0.1, 0.5, 0.9]
n = 5
p = 2

P = random.randn(n, n)
P = P @ P.T
E = random.randn(n, p)
E = E @ E.T
g = random.randn(n, 1)
D = diag(g.ravel())

Pg = P @ g
Px1 = P @ g
m = 0.5 * (g.T @ Px1)
xoE = g * E
PxoE = P @ xoE
ETxPxE = 0.5 * (xoE.T @ PxoE)
ETxPx1 = xoE.T @ Px1
ETxPx11xPxE = 0.25 / m * (ETxPx1 @ ETxPx1.T)
ZTIminusMZ = ETxPxE - ETxPx11xPxE
eigh = eigvalsh(ZTIminusMZ)
print(sorted(list(eigh)))

eta = ETxPx11xPxE @ ZTIminusMZ
vareta = 4 * trace(eta)

OneZTZE = 0.5 * (g.T @ PxoE)
tau_top = OneZTZE @ OneZTZE.T
tau_rho = empty(len(rhos))
for i in range(len(rhos)):
    tau_rho[i] = rhos[i] * m + (1 - rhos[i]) / m * tau_top

MuQ = sum(eigh)
VarQ = sum(eigh ** 2) * 2 + vareta
KerQ = sum(eigh ** 4) / (sum(eigh ** 2) ** 2) * 12
Df = 12 / KerQ
breakpoint()

# ------------------------------------------------------ #

Pg = P @ g
m = (g.T @ Pg)[0, 0]
M = 1 / m * (sqrtm(P) @ g @ g.T @ sqrtm(P))
H1 = E.T @ D.T @ P @ D @ E
H2 = E.T @ D.T @ sqrtm(P) @ M @ sqrtm(P) @ D @ E
H = H1 - H2
lambdas = eigvalsh(H / 2)
nus = chisquared(dof=1, n=s)
phi = sum(lambdas * nus)
v = sqrtm(P) @ y
xi = 2 * v.T @ (I - M) @ Z @ E @ E.T @ Z.T @ M @ v
kapa = phi + xi
tau_rho = (1 - rho) * m + rho / m * (g.T @ P @ diag(g) @ E @ E.T @ diag(g) @ P @ g)
kapas = 0.5 * rho * kapa
Q_rho = sum(tau_rho, kapas)

print(sorted(list(lambdas)))

# nus = chi(s)
#
#
