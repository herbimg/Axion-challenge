import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

import pandas as pd

def load_data():
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=500, stratify=y, random_state=0)

    scaler = MinMaxScaler(feature_range=(0,np.pi))
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    X_te = np.clip(X_te, 0, np.pi)

    y_tr = 2*y_tr -1
    y_te = 2*y_te -1
    return X_tr, X_te, y_tr, y_te

VARIANTS = ["A", "B", "C", "D", "A-ZZ"]

def build_circuit(variant):
    x  = ParameterVector("x", 2) # features
    th = ParameterVector("t", 4) # parameters θ
    qc = QuantumCircuit(2)

    # 01 encoding: x[0] na kubit 0, x[1] na kubit 1 (slajd 35)
    # 02 ansatz: pierwsza warstwa RY(t[0]), RY(t[1])

    if variant == "B":
        qc.ry(x[0], 0);
        qc.ry(x[1], 1)  # 01 encoding

        qc.ry(th[0], 0);
        qc.ry(th[1], 1)  # 02 ansatz: rotations

        qc.cx(0, 1)  # entanglement

        qc.ry(th[2], 0);
        qc.ry(th[3], 1)


    elif variant == "C":
        qc.ry(x[0], 0);
        qc.ry(x[1], 1)  # 01 encoding

        qc.ry(th[0], 0);
        qc.ry(th[1], 1)  # 02 ansatz: rotations

        qc.cz(0, 1)  # entanglement

        qc.ry(th[2], 0);
        qc.ry(th[3], 1)

    elif variant == "D":
        qc.ry(x[0], 0);
        qc.ry(x[1], 1)  # 01 encoding

        qc.ry(th[0], 0);
        qc.ry(th[1], 1)  # 02 ansatz: rotations

        qc.cx(0, 1)  # entanglement
        qc.cz(0, 1)  # entanglement

        qc.ry(th[2], 0);
        qc.ry(th[3], 1)
    elif variant == "CNOT2":
        qc.ry(x[0], 0);
        qc.ry(x[1], 1)  # 01 encoding

        qc.ry(th[0], 0);
        qc.ry(th[1], 1)  # 02 ansatz: rotations

        qc.cx(0, 1)  # entanglement
        qc.cx(0, 1)  # entanglement

        qc.ry(th[2], 0);
        qc.ry(th[3], 1)
    # A i A-ZZ: brak entanglementu
    else:
        qc.ry(x[0], 0);
        qc.ry(x[1], 1)  # 01 encoding

        qc.ry(th[0], 0);
        qc.ry(th[1], 1)  # 02 ansatz: rotations

        qc.ry(th[2], 0);
        qc.ry(th[3], 1)

    # druga warstwa RY(t[2]), RY(t[3])

    if variant == "A-ZZ":
        obs = SparsePauliOp.from_sparse_list([("ZZ", [0, 1], 1.0)], num_qubits=2)
    else:
        obs = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=2)
    return qc, obs


estimator = StatevectorEstimator()

def predict(variant, X, th):
    qc, obs = build_circuit(variant)
    M = len(X)
    params = np.hstack([np.tile(th, (M, 1)), X])
    assert params.shape == (M, 6)
    res = estimator.run([(qc, obs, params)]).result()[0]
    return res.data.evs                         # kształt (M,)

def loss_and_acc(variant, X, y, th):
    f = predict(variant, X, th)
    loss = np.mean((f - y)**2)
    acc = np.mean(np.sign(f) == y)
    return loss, acc

def grad_L(variant, X, y, th):
    f = predict(variant, X, th)
    g = np.zeros(4)
    for j in range(4):
        shift = np.zeros(4)
        shift[j] = np.pi /2
        df = (predict(variant, X, th + shift) - predict(variant, X, th - shift))/2
        g[j] = np.mean(2*(f - y) * df)
    return g

def train(variant, seed, X_tr, y_tr, lr=0.1, epochs=100):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2*np.pi, 4)
    history = []
    for epoch in range(epochs):
        loss, acc = loss_and_acc(variant, X_tr, y_tr, th)
        history.append((loss, acc))
        th = th - grad_L(variant, X_tr, y_tr, th)*lr
    return th, history

if __name__ == "__main__":
    # X_tr, X_te, y_tr, y_te = load_data()
    # print(X_tr.shape, X_te.shape)
    # print(X_tr.min(axis=0), X_tr.max(axis=0))
    # print(X_te.min(), X_te.max())
    # print(np.unique(y_tr, return_counts=True))

    # for v in VARIANTS + ["CNOT2"]:
    #     qc, obs = build_circuit(v)
    #     print(v, obs)
    #     print(qc.draw())
    # print(qc.parameters)

    # X = np.array([[1.0, 2.0]])
    #
    # th = np.array([0.3, 0.5, 0.8, 1.1])
    # for v in VARIANTS + ["CNOT2"]:
    #     print(v, predict(v, X, th))

    # X_tr, X_te, y_tr, y_te = load_data()
    # th = np.array([0.3, 0.5, 0.8, 1.1])
    # for v in VARIANTS:
    #     print(v, loss_and_acc(v, X_tr, y_tr, th), np.round(grad_L(v, X_tr, y_tr, th), 4))

    # X_tr, X_te, y_tr, y_te = load_data()
    # print(np.random.default_rng(0).uniform(0, 2 * np.pi, 4))
    # t0 = time.perf_counter()
    # th, hist = train("B", 0, X_tr, y_tr, epochs=11)
    # print("czas na epokę:", (time.perf_counter() - t0) / 11, "s")
    # print("epoka 0:", hist[0])
    # print("epoka 10:", hist[10])

    X_tr, X_te, y_tr, y_te = load_data()
    rows, curves = [], []
    for v in VARIANTS:
        for seed in range(5):
            th, hist = train(v, seed, X_tr, y_tr)
            train_loss, train_acc = loss_and_acc(v, X_tr, y_tr, th)
            test_loss, test_acc = loss_and_acc(v, X_te, y_te, th)
            rows.append({"variant": v, "seed": seed,
                         "train_acc": train_acc, "test_acc": test_acc,
                         "train_loss": train_loss, "th": th.round(4).tolist()})
            for epoch, (loss, acc) in enumerate(hist):
                curves.append({"variant": v,
                               "seed": seed,
                               "epoch": epoch,
                               "loss": loss,
                               "acc": acc})
            pd.DataFrame(rows).to_csv("results.csv", index=False)
            pd.DataFrame(curves).to_csv("curves.csv", index=False)
            print(v, seed, "test acc:", round(test_acc, 3))