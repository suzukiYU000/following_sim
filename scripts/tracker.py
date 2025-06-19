"""Kalman filter based person tracker."""

import numpy as np


class KalmanTrack:
    def __init__(self, pid: int, init_pos, dt: float = 0.1):
        self.id = pid
        self.dt = dt
        self.x = np.array([init_pos[0], init_pos[1], 0.0, 0.0])
        self.P = np.eye(4)
        self.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        self.B = np.array([[dt, 0], [0, dt], [0, 0], [0, 0]])
        self.Q = np.eye(4) * 0.01
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = np.eye(2) * 0.1
        self.misses = 0

    def predict(self, u):
        self.x = self.F.dot(self.x) + self.B.dot(u)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.x[:2]

    def update(self, z):
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x += K.dot(y)
        self.P = (np.eye(4) - K.dot(self.H)).dot(self.P)
        self.misses = 0

    def miss(self):
        self.misses += 1
