"""Person model for the simulation."""

from __future__ import annotations

import numpy as np


class Person:
    """Simple agent with obstacle avoidance behavior."""

    def __init__(self, pid: int, x: float, y: float):
        self.id = pid
        self.pos = np.array([x, y], dtype=float)
        angle = np.random.uniform(0, 2 * np.pi)
        speed = 0.5
        self.u = np.array([np.cos(angle) * speed, np.sin(angle) * speed])
        self.radius = 0.3

    def update(
        self,
        dt: float,
        persons: list["Person"],
        obstacles,
        walls,
        robot_pos=None,
        robot_radius: float = 0.4,
    ):
        rep = np.zeros(2)
        # avoid other persons
        for other in persons:
            if other.id == self.id:
                continue
            diff = self.pos - other.pos
            d = np.linalg.norm(diff)
            desired = self.radius + other.radius
            if d < desired and d > 1e-3:
                rep += (diff / d) * (desired - d) * 2.0
        # avoid obstacles (tables)
        for obs_min, obs_max in obstacles:
            closest = np.clip(self.pos, obs_min, obs_max)
            diff = self.pos - closest
            d = np.linalg.norm(diff)
            desired = self.radius + 1  # safe margin
            if d < desired and d > 1e-3:
                rep += (diff / d) * (desired - d) * 3.0
        # avoid robot
        if robot_pos is not None:
            diff = self.pos - robot_pos
            d = np.linalg.norm(diff)
            desired = self.radius + robot_radius
            if d < desired and d > 1e-3:
                rep += (diff / d) * (desired - d) * 2.0

        # avoid walls
        xmin, xmax, ymin, ymax = walls
        dist_left = self.pos[0] - xmin
        if dist_left < self.radius:
            rep[0] += (self.radius - dist_left) * 3.0
        dist_right = xmax - self.pos[0]
        if dist_right < self.radius:
            rep[0] -= (self.radius - dist_right) * 3.0
        dist_bot = self.pos[1] - ymin
        if dist_bot < self.radius:
            rep[1] += (self.radius - dist_bot) * 3.0
        dist_top = ymax - self.pos[1]
        if dist_top < self.radius:
            rep[1] -= (self.radius - dist_top) * 3.0

        stop = False
        if robot_pos is not None:
            ahead = robot_pos - self.pos
            dist = np.linalg.norm(ahead)
            if dist < 1.0:
                heading = self.u / (np.linalg.norm(self.u) + 1e-6)
                if np.dot(heading, ahead / (dist + 1e-6)) > 0.8:
                    stop = True

        if stop:
            move = rep * dt
        else:
            move = (self.u + rep) * dt
            self.u += rep * dt
            speed = np.linalg.norm(self.u)
            if speed > 1.0:
                self.u = self.u / speed

        self.pos += move
        speed = np.linalg.norm(self.u)
        if speed > 1.0:
            self.u = self.u / speed
