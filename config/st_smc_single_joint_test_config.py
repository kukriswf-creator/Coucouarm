from pathlib import Path
import numpy as np

N_JOINT = 6
PROJECT_DIR = Path("/home/liu/coucouarm_v5/ros2_workspace")
URDF_PATH = PROJECT_DIR / "coucouarm_v5_urdf" / "urdf" / "coucouarm_v5.urdf"
CSV_DIR = PROJECT_DIR / "data" / "csv"

JOINT_NAMES = [f"joint_{i + 1}" for i in range(N_JOINT)]
JOINT_STATE_TOPIC = "/joint_states"
COMMAND_TOPIC = "/arm_MIT_command_controller/commands"

JOINT_MOVES = [
    (1, 60.0),
    (1, -60.0),
    (2, 30.0),
    (2, -30.0),
    (3, 30.0),
    (4, 90.0),
    (4, -90.0),
    (5, 40.0),
    (5, -40.0),
    (6, 90.0),
    (6, -90.0),
]

HOME_DEGREES = [0.0, 30.0, -30.0, -90.0, 0.0, 0.0]

HOME_MOVE_DURATION = 3.0
POSITION_TOLERANCE_DEG = 2.0
MOVE_DURATION = 2.0
HOLD_DURATION = 0.5
RATE_HZ = 50.0
MAX_ABS_DEGREES = 100.0

G_SCALE = np.array([1.0, 1.5, 1.5, 1.0, 1.2, 1.2], dtype=float)

alpha = np.array([11.0, 15.0, 15.0, 5.0, 7.0, 5.0], dtype=float)
k1 = np.array([12.0, 12.0, 15.5, 6.5, 4.0, 4.0], dtype=float)
k2 = np.array([7.0, 7.0, 8.0, 5.0, 3.5, 3.0], dtype=float)

phi = np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.02], dtype=float)

S_DEAD = np.array([0.010, 0.012, 0.015, 0.015, 0.015, 0.040], dtype=float)
S_REENABLE = 1.5 * S_DEAD
NU_LEAK = np.array([0.8, 0.8, 0.8, 1.0, 1.0, 1.2], dtype=float)
NU_LIMIT = np.array([8.0, 10.0, 10.0, 6.0, 5.0, 4.0], dtype=float)
