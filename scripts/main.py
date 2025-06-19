"""Run the person following simulation."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment

from .person import Person
from .sensor import Sensor
from .tracker import KalmanTrack
from .robot import Robot


def run():
    np.random.seed(0)
    walls = (-9.5, 9.5, -9.5, 9.5)
    table = (np.array([3, -1]), np.array([5, 1]))
    obstacles = [table]
    persons = [Person(i, np.random.uniform(-5, 5), np.random.uniform(-5, 5)) for i in range(5)]
    sensor = Sensor(noise_std=0.2)
    tracks = []
    robot = Robot(0, 0, np.pi / 2, stop_dist=1.0)
    dt = 0.1

    fig, ax = plt.subplots()
    ax.set_xlim(walls[0], walls[1])
    ax.set_ylim(walls[2], walls[3])
    ax.set_aspect("equal")
    ax.plot([walls[0], walls[1], walls[1], walls[0], walls[0]], [walls[2], walls[2], walls[3], walls[3], walls[2]], "k-")
    tab = plt.Rectangle(table[0], table[1][0] - table[0][0], table[1][1] - table[0][1], fc="sienna", ec="black")
    ax.add_patch(tab)
    person_plots = [ax.plot([], [], "o", markersize=8)[0] for _ in persons]
    robot.add_to_ax(ax)
    follower_text = ax.text(walls[0] + 0.5, walls[3] - 0.5, "", fontsize=12)

    def init():
        return person_plots + [robot.patch, follower_text]

    def animate(_):
        nonlocal tracks
        for p, pp in zip(persons, person_plots):
            p.update(dt, persons, obstacles, walls)
            pp.set_data([p.pos[0]], [p.pos[1]])
        meas_list = sensor.scan(persons, robot.pose[:2], obstacles, walls)
        preds = [t.predict(persons[t.id].u) for t in tracks]
        if preds and meas_list:
            positions = [m["pos"] for m in meas_list]
            cost = np.linalg.norm(np.expand_dims(preds, 1) - np.expand_dims(positions, 0), axis=2)
            row, col = linear_sum_assignment(cost)
            assigned_t, assigned_m = set(), set()
            for ti, mj in zip(row, col):
                tracks[ti].update(positions[mj])
                assigned_t.add(ti)
                assigned_m.add(mj)
            for i in range(len(tracks)):
                if i not in assigned_t:
                    tracks[i].miss()
            for mj, m in enumerate(meas_list):
                if mj not in assigned_m:
                    tracks.append(KalmanTrack(m["id"], m["pos"], dt))
            tracks[:] = [t for t in tracks if t.misses < 5]
        else:
            for m in meas_list:
                tracks.append(KalmanTrack(m["id"], m["pos"], dt))
        for line in ax.lines[len(person_plots) + 1 :]:
            line.remove()
        for t in tracks:
            ax.plot([t.x[0]], [t.x[1]], "gx")
        follower = max(persons, key=lambda p: p.pos[1])
        follower_text.set_text(f"Following: P{follower.id}")
        for p, pp in zip(persons, person_plots):
            pp.set_markerfacecolor("red" if p.id == follower.id else "blue")
        ttrack = next((t for t in tracks if t.id == follower.id), None)
        if ttrack:
            robot.move_towards(ttrack.x[:2], persons[follower.id].u, dt, persons, obstacles, walls)
        robot.update_patch()
        return person_plots + [robot.patch, follower_text]

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=100, blit=True)
    plt.show()


if __name__ == "__main__":
    run()
