"""Sensor model for detecting persons."""

from __future__ import annotations

import numpy as np

from .utils import segment_intersects_rect


class Sensor:
    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std

    def scan(self, persons, sensor_pos, obstacles, walls):
        measurements = []
        xmin, xmax, ymin, ymax = walls
        for p in persons:
            if not (xmin <= p.pos[0] <= xmax and ymin <= p.pos[1] <= ymax):
                continue
            occluded = any(
                segment_intersects_rect(sensor_pos, p.pos, obs_min, obs_max)
                for obs_min, obs_max in obstacles
            )
            if occluded:
                continue
            noisy = p.pos + np.random.normal(0, self.noise_std, 2)
            measurements.append({"id": p.id, "pos": noisy})
        return measurements
