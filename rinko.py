import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment

# --- Utility: Ray-Rectangle Intersection (Liang-Barsky) ---
def segment_intersects_rect(p1, p2, rect_min, rect_max):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    p = [-dx, dx, -dy, dy]
    q = [p1[0] - rect_min[0], rect_max[0] - p1[0], p1[1] - rect_min[1], rect_max[1] - p1[1]]
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-6:
            if qi < 0: return False
        else:
            t = qi / pi
            if pi < 0:
                if t > u1: return False
                if t > u0: u0 = t
            else:
                if t < u0: return False
                if t < u1: u1 = t
    return True

# --- Person Model (Wheelchair with Control Input + Enhanced Avoidance) ---
class Person:
    def __init__(self, id, x, y):
        self.id = id
        self.pos = np.array([x, y], dtype=float)
        angle = np.random.uniform(0, 2*np.pi)
        speed = 0.5
        self.u = np.array([np.cos(angle)*speed, np.sin(angle)*speed])
        self.radius = 0.3

    def update(self, dt, persons, obstacles, walls):
        rep = np.zeros(2)
        # avoid other persons
        for other in persons:
            if other.id == self.id: continue
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
        # avoid walls
        xmin, xmax, ymin, ymax = walls
        # left wall
        dist_left = self.pos[0] - xmin
        if dist_left < self.radius:
            rep[0] += (self.radius - dist_left) * 3.0
        # right wall
        dist_right = xmax - self.pos[0]
        if dist_right < self.radius:
            rep[0] -= (self.radius - dist_right) * 3.0
        # bottom wall
        dist_bot = self.pos[1] - ymin
        if dist_bot < self.radius:
            rep[1] += (self.radius - dist_bot) * 3.0
        # top wall
        dist_top = ymax - self.pos[1]
        if dist_top < self.radius:
            rep[1] -= (self.radius - dist_top) * 3.0
        # apply movement
        self.pos += (self.u + rep) * dt
        self.u += rep * dt
        # limit speed
        speed = np.linalg.norm(self.u)
        if speed > 1.0:
            self.u = self.u / speed

# --- Sensor Simulation with Occlusion ---
class Sensor:
    def __init__(self, noise_std=0.05):
        self.noise_std = noise_std

    def scan(self, persons, sensor_pos, obstacles, walls):
        measurements = []
        xmin, xmax, ymin, ymax = walls
        for p in persons:
            # within walls
            if not (xmin <= p.pos[0] <= xmax and ymin <= p.pos[1] <= ymax):
                continue
            # occlusion by obstacles
            occluded = any(segment_intersects_rect(sensor_pos, p.pos, obs_min, obs_max)
                           for obs_min, obs_max in obstacles)
            if occluded: continue
            noisy = p.pos + np.random.normal(0, self.noise_std, 2)
            measurements.append({'id': p.id, 'pos': noisy})
        return measurements

# --- Kalman Filter Track with Control Input ---
class KalmanTrack:
    def __init__(self, id, init_pos, dt=0.1):
        self.id = id
        self.dt = dt
        self.x = np.array([init_pos[0], init_pos[1], 0., 0.])
        self.P = np.eye(4)
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self.B = np.array([[dt,0],[0,dt],[0,0],[0,0]])
        self.Q = np.eye(4)*0.01
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.R = np.eye(2)*0.1
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

# --- Robot (Differential Drive + Avoidance) ---
class Robot:
    def __init__(self, x, y, theta, kp_dist=1.0, kp_angle=2.0, stop_dist=1.0):
        self.pose = np.array([x, y, theta])
        self.kp_dist = kp_dist
        self.kp_angle = kp_angle
        self.stop_dist = stop_dist
        self.radius = 0.4
        self.patch = plt.Rectangle((0,0),0.6,0.4,fc='red',ec='black')

    def add_to_ax(self, ax):
        ax.add_patch(self.patch)

    def move_towards(self, target_pos, target_vel, dt, persons, obstacles, walls):
        dx, dy = target_pos - self.pose[:2]
        dist = np.hypot(dx, dy)
        angle_to = np.arctan2(dy, dx)
        err = (angle_to - self.pose[2] + np.pi) % (2*np.pi) - np.pi
        if dist > self.stop_dist:
            v_des = self.kp_dist * (dist - self.stop_dist) + np.linalg.norm(target_vel)
        else:
            v_des = 0.0
        rep = np.zeros(2)
        # avoid persons
        for p in persons:
            diff = self.pose[:2] - p.pos
            d = np.linalg.norm(diff)
            if d < self.radius + p.radius and d > 1e-3:
                rep += (diff / d) * ((self.radius + p.radius) - d) * 2.0
        # avoid obstacles
        for obs_min, obs_max in obstacles:
            closest = np.clip(self.pose[:2], obs_min, obs_max)
            diff = self.pose[:2] - closest
            d = np.linalg.norm(diff)
            desired = self.radius + 0.5
            if d < desired and d > 1e-6:
                rep += (diff / d) * (desired - d) * 3.0
        # avoid walls
        xmin, xmax, ymin, ymax = walls
        # left wall
        if self.pose[0] - xmin < self.radius:
            rep[0] += (self.radius - (self.pose[0] - xmin)) * 3.0
        # right wall
        if xmax - self.pose[0] < self.radius:
            rep[0] -= (self.radius - (xmax - self.pose[0])) * 3.0
        # bottom wall
        if self.pose[1] - ymin < self.radius:
            rep[1] += (self.radius - (self.pose[1] - ymin)) * 3.0
        # top wall
        if ymax - self.pose[1] < self.radius:
            rep[1] -= (self.radius - (ymax - self.pose[1])) * 3.0
        # update
        omega = self.kp_angle * err
        self.pose[2] += omega * dt
        move_vec = np.array([np.cos(self.pose[2]), np.sin(self.pose[2])]) * v_des + rep
        self.pose[:2] += move_vec * dt

    def update_patch(self):
        w, h = 0.6, 0.4
        corners = np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]])
        Rmat = np.array([[np.cos(self.pose[2]), -np.sin(self.pose[2])],
                         [np.sin(self.pose[2]),  np.cos(self.pose[2])]])
        rot = (Rmat.dot(corners.T)).T + self.pose[:2]
        self.patch.set_xy(rot[0])
        self.patch.angle = np.degrees(self.pose[2])

# --- Simulation Setup ---
np.random.seed(0)
walls = (-9.5, 9.5, -9.5, 9.5)
# table obstacle
table = (np.array([3, -1]), np.array([5, 1]))
obstacles = [table]
persons = [Person(i, np.random.uniform(-5,5), np.random.uniform(-5,5)) for i in range(5)]
sensor = Sensor(noise_std=0.2)
tracks = []
robot = Robot(0, 0, np.pi/2, stop_dist=1.0)
dt = 0.1

# Visualization
fig, ax = plt.subplots()
ax.set_xlim(walls[0], walls[1])
ax.set_ylim(walls[2], walls[3])
ax.set_aspect('equal')
# draw walls
ax.plot([walls[0],walls[1],walls[1],walls[0],walls[0]], [walls[2],walls[2],walls[3],walls[3],walls[2]], 'k-')
# draw table
tab = plt.Rectangle(table[0], table[1][0]-table[0][0], table[1][1]-table[0][1], fc='sienna', ec='black')
ax.add_patch(tab)
person_plots = [ax.plot([], [], 'o', markersize=8)[0] for _ in persons]
robot.add_to_ax(ax)
follower_text = ax.text(walls[0]+0.5, walls[3]-0.5, '', fontsize=12)

# Animation init and update

def init():
    return person_plots + [robot.patch, follower_text]

def animate(frame):
    global tracks
    # update persons
    for p, pp in zip(persons, person_plots):
        p.update(dt, persons, obstacles, walls)
        pp.set_data([p.pos[0]], [p.pos[1]])
    # sensor with occlusion
    meas_list = sensor.scan(persons, robot.pose[:2], obstacles, walls)
    # predict
    preds = [t.predict(persons[t.id].u) for t in tracks]
    # associate
    if preds and meas_list:
        positions = [m['pos'] for m in meas_list]
        cost = np.linalg.norm(np.expand_dims(preds,1) - np.expand_dims(positions,0), axis=2)
        row, col = linear_sum_assignment(cost)
        assigned_t, assigned_m = set(), set()
        for ti, mj in zip(row, col):
            tracks[ti].update(positions[mj]); assigned_t.add(ti); assigned_m.add(mj)
        for i in range(len(tracks)):
            if i not in assigned_t: tracks[i].miss()
        for mj, m in enumerate(meas_list):
            if mj not in assigned_m:
                tracks.append(KalmanTrack(m['id'], m['pos'], dt))
        tracks = [t for t in tracks if t.misses < 5]
    else:
        for m in meas_list:
            tracks.append(KalmanTrack(m['id'], m['pos'], dt))
    # draw tracks
    for line in ax.lines[len(person_plots)+1:]: line.remove()
    for t in tracks: ax.plot([t.x[0]], [t.x[1]], 'gx')
    # select follower
    follower = max(persons, key=lambda p: p.pos[1])
    follower_text.set_text(f'Following: P{follower.id}')
    for p, pp in zip(persons, person_plots): pp.set_markerfacecolor('red' if p.id==follower.id else 'blue')
    # robot move
    ttrack = next((t for t in tracks if t.id==follower.id), None)
    if ttrack:
        robot.move_towards(ttrack.x[:2], persons[follower.id].u, dt, persons, obstacles, walls)
    robot.update_patch()
    return person_plots + [robot.patch, follower_text]

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=100, blit=True)
plt.show()
