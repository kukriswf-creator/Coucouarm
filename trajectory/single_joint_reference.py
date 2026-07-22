import numpy as np

def quintic_reference(q_start,q_end,ratio,duration):

    q_start = np.asarray(q_start)
    q_end = np.asarray(q_end)

    r = float(np.clip(ratio, 0.0, 1.0))
    delta = q_end - q_start

    smooth = 10.0 * r**3 - 15.0 * r**4 + 6.0 * r**5

    smooth_dot = (
        30.0 * r**2 - 60.0 * r**3 + 30.0 * r**4
    ) / duration

    smooth_ddot = (
        60.0 * r - 180.0 * r**2 + 120.0 * r**3
    ) / duration**2

    q_des = q_start + delta * smooth
    dq_des = delta * smooth_dot
    ddq_des = delta * smooth_ddot
    return q_des, dq_des, ddq_des

def hold_reference(q_des):
    q_des = np.asarray(q_des)
    dq_des = np.zeros(6)
    ddq_des= np.zeros(6)
    return q_des.copy(), dq_des.copy(), ddq_des.copy()
