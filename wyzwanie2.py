import numpy as np


H = 1/np.sqrt(2) * np.array([[1, 1],
             [1, -1]],dtype=complex)
X = np.array([[0, 1],
              [1, 0]],dtype=complex)
Z = np.array([[1, 0],
              [0, -1]],dtype=complex)
I = np.eye(2, dtype=complex)
def ry(kat ):
    return np.array([[np.cos(kat/2), -1 * np.sin(kat/2)],
                     [np.sin(kat/2), np.cos(kat/2)]], dtype=complex)

# kontrolny do cnot po lewej

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)

def stan_zero(n : int):
    # n - ilosc kubitow
    #stan zero (ket 0) |0>
    # [1]
    # [0]
    x = np.zeros(2**n, dtype=complex)
    x[0] = 1
    return x


a = 1
b = 0
amp = np.array([a, b], dtype=complex)

ket0 = stan_zero(1)
ket1 = X @ stan_zero(1)
psi = amp[0] * ket0 + amp[1] * ket1
psi2 = np.kron(ket0, ket0)

print(psi)
print(psi2)

def measure(psi, shots : int):

    probs = (np.abs(psi))**2 # reguła borna
    n = int(np.log2(len(psi)))
    outcomes = np.random.choice(len(psi), size=shots, p=probs)
    counts = {}
    for o in outcomes:
        key = format(o, f"0{n}b")  # indeks → napis bitowy, np. 3 → "11"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))

print(H @ stan_zero(1))
print(H @ ( H @stan_zero(1)))
# działa H·H|0⟩ = |0⟩

#teraz robimy bella
psi2 = CNOT @ np.kron(H, I) @ psi2
print(psi2)
# ok to działa :D
# to teraz sprawdzenie

print(measure(psi2, shots=1000))



