import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class ICA:
    def __init__(self, nsource=2, nspeaker=1024):
        self.nsource = nsource
        self.nspeaker = nspeaker
        self.W = np.random.randn(nsource, nspeaker)

    def whiten(self, X):
        X = X - X.mean(axis=1, keepdims=True)

        cov = np.cov(X)
        d, E = np.linalg.eigh(cov)

        D_inv = np.diag(1.0 / np.sqrt(d + 1e-8))
        X_white = E @ D_inv @ E.T @ X
        return X_white

    def train(self, X, lr=1e-3, epochs=20):
        # X: (examples, frames, features)
        X = X.reshape(-1, self.nspeaker).T  # (features, samples)
        X = self.whiten(X)

        for _ in range(epochs):
            Y = self.W @ X

            # nonlinearity (tanh is standard ICA choice)
            g = np.tanh(Y)
            g_der = 1 - g**2

            dW = (np.eye(self.nsource) + (g @ Y.T) / X.shape[1]) @ self.W

            self.W += lr * dW

    def predict(self, X):
        # (n_frames, nspeaker)
        S = self.W @ X.T   # (nsource, n_frames)

        return S.T   # (n_frames, nsource)
