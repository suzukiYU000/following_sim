"""Run the person following simulation."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment
import yaml

from .person import Person
from .sensor import Sensor
from .tracker import KalmanTrack
from .robot import Robot


def run(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    np.random.seed(cfg.get("random_seed", 0))
    walls = tuple(cfg.get("walls", [-9.5, 9.5, -9.5, 9.5]))
    table_cfg = cfg.get("table", [[3, -1], [5, 1]])
    table = (np.array(table_cfg[0]), np.array(table_cfg[1]))
    obstacles = [table]

    pcfg = cfg.get("person", {})
    num_persons = cfg.get("num_persons", 5)
    persons = [
        Person(
            i,
            np.random.uniform(-5, 5),
            np.random.uniform(-5, 5),
            speed=pcfg.get("speed", 0.5),
            radius=pcfg.get("radius", 0.3),
        )
        for i in range(num_persons)
    ]

    sensor = Sensor(noise_std=cfg.get("sensor", {}).get("noise_std", 0.2))
    tracks = []

    rcfg = cfg.get("robot", {})
    robot = Robot(
        rcfg.get("x", 0),
        rcfg.get("y", 0),
        rcfg.get("theta", np.pi / 2),
        kp_dist=rcfg.get("kp_dist", 1.0),
        kp_angle=rcfg.get("kp_angle", 2.0),
        stop_dist=rcfg.get("stop_dist", 1.0),
    )

    dt = cfg.get("sim", {}).get("dt", 0.1)
    follower_id = None

    fig, ax = plt.subplots()
    ax.set_xlim(walls[0], walls[1])
    ax.set_ylim(walls[2], walls[3])
    ax.set_aspect("equal")
    ax.plot([walls[0], walls[1], walls[1], walls[0], walls[0]], [walls[2], walls[2], walls[3], walls[3], walls[2]], "k-")
    tab = plt.Rectangle(table[0], table[1][0] - table[0][0], table[1][1] - table[0][1], fc="sienna", ec="black")
    ax.add_patch(tab)
    person_plots = [ax.plot([], [], "o", markersize=8)[0] for _ in persons]
    robot.add_to_ax(ax)
    follower_text = ax.text(
        walls[0] + 0.5, walls[3] - 0.5, "Click a person to follow", fontsize=12
    )

    for p, pp in zip(persons, person_plots):
        pp.set_data([p.pos[0]], [p.pos[1]])

    def on_click(event):
        nonlocal follower_id
        if event.inaxes != ax or follower_id is not None:
            return
        click = np.array([event.xdata, event.ydata])
        dists = [np.linalg.norm(p.pos - click) for p in persons]
        follower_id = persons[int(np.argmin(dists))].id
        follower_text.set_text(f"Following: P{follower_id}")
        for p, pp in zip(persons, person_plots):
            pp.set_markerfacecolor("red" if p.id == follower_id else "blue")

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    while follower_id is None:
        plt.pause(0.1)
    fig.canvas.mpl_disconnect(cid)

    def init():
        return person_plots + [robot.patch, follower_text]

    def animate(_):
        nonlocal tracks
        for p, pp in zip(persons, person_plots):
            p.update(dt, persons, obstacles, walls, robot.pose[:2], robot.radius)
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
        follower = next((p for p in persons if p.id == follower_id), None)
        if follower:
            follower_text.set_text(f"Following: P{follower.id}")
            for p, pp in zip(persons, person_plots):
                pp.set_markerfacecolor("red" if p.id == follower.id else "blue")
            ttrack = next((t for t in tracks if t.id == follower.id), None)
            if ttrack:
                robot.move_towards(
                    ttrack.x[:2], persons[follower.id].u, dt, persons, obstacles, walls
                )
        robot.update_patch()
        return person_plots + [robot.patch, follower_text]

    sim_cfg = cfg.get("sim", {})
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=sim_cfg.get("frames", 200),
        interval=sim_cfg.get("interval", 100),
        blit=True,
    )
    plt.show()


if __name__ == "__main__":
    run()
