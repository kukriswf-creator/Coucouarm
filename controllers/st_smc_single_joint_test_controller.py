import time
import numpy as np
from config import st_smc_single_joint_test_config as cfg
from dynamics.dynamic import RobotDynamics

class STSMCController:

    def __init__(self, dynamics: RobotDynamics):
        self.dynamics = dynamics
        self.nu = np.zeros(cfg.N_JOINT, dtype=float)

        self.nu_integration_enabled = np.ones(cfg.N_JOINT, dtype=bool)
        self.last_control_time = None

    def reset_control_clock(self):
        self.last_control_time = None

    def reset_state(self): 
        self.nu.fill(0.0)
        self.nu_integration_enabled.fill(True)
        self.reset_control_clock()


    def _get_control_dt(self):
        now = time.time()
        if self.last_control_time is None:
            dt = 1.0 / cfg.RATE_HZ
        else:
            dt = now - self.last_control_time

        self.last_control_time = now
        return dt

    def _st_reach_term(self, s, dt):
       
        s = np.asarray(s)
        sigma = np.clip(s / cfg.phi, -1.0, 1.0)
        abs_s = np.abs(s)

        self.nu_integration_enabled[abs_s <= cfg.S_DEAD] = False
        self.nu_integration_enabled[abs_s >= cfg.S_REENABLE] = True
        active = self.nu_integration_enabled

        nu_dot = -cfg.NU_LEAK * self.nu

        nu_dot[active] += -cfg.k2[active] * np.sign(s[active])
        nu_dot[(self.nu >= cfg.NU_LIMIT) & (nu_dot > 0.0)] = 0.0
        nu_dot[(self.nu <= -cfg.NU_LIMIT) & (nu_dot < 0.0)] = 0.0

        self.nu += nu_dot * dt
        self.nu = np.clip(self.nu, -cfg.NU_LIMIT, cfg.NU_LIMIT)

        u_st = (
            -cfg.k1 * np.sqrt(np.abs(s) + 1e-8) * sigma
            + self.nu
        )
        return u_st

    def compute(self,q,dq,q_des,dq_des,ddq_des,):

        q = np.asarray(q)
        dq = np.asarray(dq)
        q_des = np.asarray(q_des)
        dq_des = np.asarray(dq_des)
        ddq_des = np.asarray(ddq_des)
        M, C, G = self.dynamics.compute(q, dq)

        e = q_des - q
        de = dq_des - dq

        s = de + cfg.alpha * e

        dt = self._get_control_dt()
        u_st = self._st_reach_term(s, dt)
        tau_eq = M @ (ddq_des + cfg.alpha * de) + C @ dq + G
        tau_sw = -M @ u_st
        tau = tau_eq + tau_sw

        return tau, s, tau_sw, u_st, G, dt
