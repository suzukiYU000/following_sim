# Following Simulation

This repository contains a minimal simulation where a robot tracks a single person among multiple moving agents.

## Features

- Simple obstacle and wall avoidance for persons and the robot.
- Kalman filter based tracking with noisy measurements.
- Select the person to follow by clicking on them when the simulation starts.

## Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

## Running

Execute the simulation with:

```bash
python main.py
```

Parameters for the simulation are stored in `config.yaml`. Modify this file to change the environment or robot behavior. A window will open showing the initial positions of all persons. Click on one of them to choose the target to follow. The robot will then attempt to follow that person while avoiding obstacles and other people.
