import numpy as np
import pinocchio as pin

class RobotDynamics:
    def __init__(self, urdf_path, g_scale, n_joint):
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.n_joint = n_joint
        self.g_scale = np.asarray(g_scale, dtype=float).reshape(n_joint)

        if self.model.nv != n_joint:
            raise ValueError(f"模型自由度错误！")

    def compute(self,q,dq):
        q = np.asarray(q)
        dq = np.asarray(dq)
        M = np.asarray(pin.crba(self.model, self.data, q), dtype=float)
        M = 0.5 * (M + M.T)

        C = np.asarray(
            pin.computeCoriolisMatrix(self.model, self.data, q, dq),
            dtype=float,
        )

        G_raw = np.asarray(
            pin.computeGeneralizedGravity(self.model, self.data, q),
            dtype=float,
        ).reshape(self.n_joint)

        G = self.g_scale * G_raw
        return M, C, G
