"""Robot model with simple obstacle avoidance."""

import numpy as np
import matplotlib.pyplot as plt


class Robot:
    def __init__(self, x, y, theta, kp_dist=1.0, kp_angle=2.0, stop_dist=1.0):
        self.pose = np.array([x, y, theta])
        self.kp_dist = kp_dist
        self.kp_angle = kp_angle
        self.stop_dist = stop_dist
        self.radius = 0.4
        self.patch = plt.Rectangle((0, 0), 0.6, 0.4, fc="red", ec="black")

    def add_to_ax(self, ax):
        ax.add_patch(self.patch)

    def move_towards(self, target_pos, target_vel, dt, persons, obstacles, walls):
        dx, dy = target_pos - self.pose[:2]
        dist = np.hypot(dx, dy)
        angle_to = np.arctan2(dy, dx)
        err = (angle_to - self.pose[2] + np.pi) % (2 * np.pi) - np.pi
        if dist > self.stop_dist:
            v_des = self.kp_dist * (dist - self.stop_dist) + np.linalg.norm(target_vel)
        else:
            v_des = 0.0
        rep = np.zeros(2)
        for p in persons:
            diff = self.pose[:2] - p.pos
            d = np.linalg.norm(diff)
            if d < self.radius + p.radius and d > 1e-3:
                rep += (diff / d) * ((self.radius + p.radius) - d) * 2.0
        for obs_min, obs_max in obstacles:
            closest = np.clip(self.pose[:2], obs_min, obs_max)
            diff = self.pose[:2] - closest
            d = np.linalg.norm(diff)
            desired = self.radius + 0.5
            if d < desired and d > 1e-6:
                rep += (diff / d) * (desired - d) * 3.0
        xmin, xmax, ymin, ymax = walls
        if self.pose[0] - xmin < self.radius:
            rep[0] += (self.radius - (self.pose[0] - xmin)) * 3.0
        if xmax - self.pose[0] < self.radius:
            rep[0] -= (self.radius - (xmax - self.pose[0])) * 3.0
        if self.pose[1] - ymin < self.radius:
            rep[1] += (self.radius - (self.pose[1] - ymin)) * 3.0
        if ymax - self.pose[1] < self.radius:
            rep[1] -= (self.radius - (ymax - self.pose[1])) * 3.0
        omega = self.kp_angle * err
        self.pose[2] += omega * dt
        move_vec = np.array([np.cos(self.pose[2]), np.sin(self.pose[2])]) * v_des + rep
        self.pose[:2] += move_vec * dt

    def update_patch(self):
        w, h = 0.6, 0.4
        corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        Rmat = np.array(
            [[np.cos(self.pose[2]), -np.sin(self.pose[2])], [np.sin(self.pose[2]), np.cos(self.pose[2])]]
        )
        rot = (Rmat.dot(corners.T)).T + self.pose[:2]
        self.patch.set_xy(rot[0])
        self.patch.angle = np.degrees(self.pose[2])
