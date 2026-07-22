#!/usr/bin/env python3
import csv
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from config import st_smc_single_joint_test_config as cfg
from controllers.st_smc_single_joint_test_controller import STSMCController
from dynamics.dynamic import RobotDynamics
from trajectory.single_joint_reference import hold_reference, quintic_reference


PHASE_HOME_MOVE = "home_move"
PHASE_HOME_HOLD = "home_hold"
PHASE_START_HOLD = "joint_start_hold"
PHASE_OUTBOUND = "joint_outbound"
PHASE_TARGET_HOLD = "joint_target_hold"
PHASE_RETURN = "joint_return"
PHASE_RETURN_HOLD = "joint_return_hold"
PHASE_NEXT_WAIT = "inter_joint_wait"
PHASE_EMERGENCY = "emergency_hold"
PHASE_DONE = "done"


class SingleJointMove(Node):
    def __init__(self):
        super().__init__("single_joint_move")
        self.current_q = {}
        self.current_dq = {}

        self.dynamics = RobotDynamics(
            cfg.URDF_PATH,
            cfg.G_SCALE,
            cfg.N_JOINT,
        )
        self.controller = STSMCController(self.dynamics)

        self.current_phase = "wait_joint_state"
        self.phase_start_time = None
        self.phase_duration = 0.0
        self.q_phase_start = None
        self.q_phase_goal = None

        self.q_home = np.deg2rad(np.asarray(cfg.HOME_DEGREES))

        self.task_index = 0
        self.joint_number = None
        self.joint_index = None
        self.move_degrees = None
        self.q_joint_start = None
        self.q_joint_target = None

        self.initialized = False
        self.finished = False

        self._open_csv_logger()

        self.subscription = self.create_subscription(
            JointState,
            cfg.JOINT_STATE_TOPIC,
            self.joint_state_callback,
            10,
        )
        self.publisher = self.create_publisher(
            Float64MultiArray,
            cfg.COMMAND_TOPIC,
            10,
        )

        self.timer = self.create_timer(
            1.0 / cfg.RATE_HZ,
            self.control_loop,
        )

    def _open_csv_logger(self):
        cfg.CSV_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = cfg.CSV_DIR / f"joint_position_{timestamp}.csv"
        self.csv_file = self.csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow(
            ["time", "dt", "phase"]
            + [f"q_des_{i + 1}" for i in range(cfg.N_JOINT)]
            + [f"q_{i + 1}" for i in range(cfg.N_JOINT)]
            + [f"error_{i + 1}" for i in range(cfg.N_JOINT)]
            + [f"tau_{i + 1}" for i in range(cfg.N_JOINT)]
            + [f"tau_sw_{i + 1}" for i in range(cfg.N_JOINT)]
            + [f"s_{i + 1}" for i in range(cfg.N_JOINT)]
            + [f"u_st_{i + 1}" for i in range(cfg.N_JOINT)]
            + [f"nu_{i + 1}" for i in range(cfg.N_JOINT)]
        )

        self.log_start_time = time.monotonic()
        self.log_row_count = 0
        self.get_logger().info(f"关节数据将保存到: {self.csv_path}")

    def joint_state_callback(self, msg):
        for name, pos, vel in zip(msg.name, msg.position, msg.velocity):
            self.current_q[name] = float(pos)
            self.current_dq[name] = float(vel)

    def get_state(self):
        q_complete = all(
            name in self.current_q
            for name in cfg.JOINT_NAMES
        )
        dq_complete = all(
            name in self.current_dq
            for name in cfg.JOINT_NAMES
        )

        if not q_complete or not dq_complete:
            return None, None

        q = np.array(
            [self.current_q[name] for name in cfg.JOINT_NAMES],
            dtype=float,
        )
        dq = np.array(
            [self.current_dq[name] for name in cfg.JOINT_NAMES],
            dtype=float,
        )
        return q, dq

    def start_phase(self, phase, q_start, q_goal, duration, now):
        self.current_phase = phase
        self.q_phase_start = np.asarray(q_start).copy()
        self.q_phase_goal = np.asarray(q_goal).copy()
        self.phase_duration = duration
        self.phase_start_time = now
        self.controller.reset_control_clock()

    def get_reference(self, now):
        move_phases = {PHASE_HOME_MOVE,PHASE_OUTBOUND,PHASE_RETURN,}

        if self.current_phase in move_phases:
            ratio = (
                now - self.phase_start_time
            ) / self.phase_duration

            return quintic_reference(
                self.q_phase_start,
                self.q_phase_goal,
                ratio,
                self.phase_duration,
            )

        return hold_reference(self.q_phase_goal)

    def phase_finished(self, now):
        
        if now - self.phase_start_time >= self.phase_duration:
            return True
        else:
            return False

    def publish_control(self, q, dq, q_des, dq_des, ddq_des):
        tau, s, tau_sw, u_st, _G, dt = self.controller.compute(
            q,
            dq,
            q_des,
            dq_des,
            ddq_des,
        )

        msg = Float64MultiArray()
        msg.data = []

        for i in range(cfg.N_JOINT):
            msg.data.extend(
                [
                    float(q_des[i]),
                    float(dq_des[i]),
                    float(tau[i]),
                ]
            )

        self.publisher.publish(msg)
        self.log_control_data(q_des, q, tau, tau_sw, s, u_st, dt)

    def log_control_data(self, q_des, q, tau, tau_sw, s, u_st, dt):
        q_des = np.asarray(q_des, dtype=float)
        q = np.asarray(q, dtype=float)
        tau = np.asarray(tau, dtype=float)
        tau_sw = np.asarray(tau_sw, dtype=float)
        s = np.asarray(s, dtype=float)
        u_st = np.asarray(u_st, dtype=float)

        error = q_des - q
        elapsed = time.monotonic() - self.log_start_time

        self.csv_writer.writerow(
            [f"{elapsed:.9f}", f"{dt:.9f}", self.current_phase]
            + [f"{value:.7f}" for value in q_des]
            + [f"{value:.7f}" for value in q]
            + [f"{value:.7f}" for value in error]
            + [f"{value:.7f}" for value in tau]
            + [f"{value:.7f}" for value in tau_sw]
            + [f"{value:.7f}" for value in s]
            + [f"{value:.7f}" for value in u_st]
            + [f"{value:.7f}" for value in self.controller.nu]
        )

        self.log_row_count += 1
        if self.log_row_count % 50 == 0:
            self.csv_file.flush()

    def initialize_experiment(self, q, now):
        self.controller.reset_state()
        self.initialized = True

        print()
        print("移动到定义的Home点")

        self.start_phase(
            PHASE_HOME_MOVE,       
            q,                      
            self.q_home,             
            cfg.HOME_MOVE_DURATION, 
            now,                     
        )

    def print_home_check(self, q):
        print()
        print("Home点到位检查")

        for i in range(cfg.N_JOINT):
            actual_deg = np.rad2deg(q[i])
            error_deg = cfg.HOME_DEGREES[i] - actual_deg
            result = (
                "通过"
                if abs(error_deg) <= cfg.POSITION_TOLERANCE_DEG
                else "未到位"
            )

            print(
                f"joint_{i + 1}："
                f"目标 {cfg.HOME_DEGREES[i]:+.2f}°，"
                f"实际 {actual_deg:+.2f}°，"
                f"误差 {error_deg:+.2f}°，"
                f"{result}"
            )

    def start_joint_task(self, now):
        if self.task_index >= len(cfg.JOINT_MOVES):
            self.current_phase = PHASE_DONE
            self.finished = True
            print()
            print("自检运动完成")
            return

        self.joint_number, self.move_degrees = (cfg.JOINT_MOVES[self.task_index])
        self.joint_index = self.joint_number - 1

        self.q_joint_start = self.q_home.copy()
        self.q_joint_target = self.q_home.copy()
        self.q_joint_target[self.joint_index] += np.deg2rad(
            self.move_degrees
        )

        print()
        print(f"========== joint_{self.joint_number} 实验 ==========")
        print(
            "起始角度："
            f"{np.rad2deg(self.q_joint_start[self.joint_index]):+.2f}°"
        )
        print(
            "目标角度："
            f"{np.rad2deg(self.q_joint_target[self.joint_index]):+.2f}°"
        )
        print(f"命令增量：{self.move_degrees:+.2f}°")

        self.start_phase(
            PHASE_START_HOLD,
            self.q_joint_start,
            self.q_joint_start,
            0.5,
            now,
        )

    def print_target_check(self, q):
        target_deg = np.rad2deg(
            self.q_joint_target[self.joint_index]
        )
        actual_deg = np.rad2deg(q[self.joint_index])
        error_deg = target_deg - actual_deg

        print()
        print(f"目标角度：{target_deg:+.2f}°")
        print(f"实际角度：{actual_deg:+.2f}°")
        print(f"目标误差：{error_deg:+.2f}°")

    def print_return_check(self, q):
        error_deg = np.rad2deg(
            q[self.joint_index]
            - self.q_joint_start[self.joint_index]
        )

        print(f"返回误差：{error_deg:+.2f}°")
        print(f"joint_{self.joint_number} 运动和返回完成")
        print("===================================================")

    def update_phase(self, q, now):
        """当前阶段时间结束后，切换到下一阶段。"""
        if not self.phase_finished(now):
            return

        if self.current_phase == PHASE_HOME_MOVE:
            self.start_phase(
                PHASE_HOME_HOLD,
                self.q_home,
                self.q_home,
                2.0,
                now,
            )

        elif self.current_phase == PHASE_HOME_HOLD:
            self.print_home_check(q)
            self.controller.reset_state()
            self.start_joint_task(now)

        elif self.current_phase == PHASE_START_HOLD:
            self.start_phase(
                PHASE_OUTBOUND,
                self.q_joint_start,
                self.q_joint_target,
                cfg.MOVE_DURATION,
                now,
            )

        elif self.current_phase == PHASE_OUTBOUND:
            self.start_phase(
                PHASE_TARGET_HOLD,
                self.q_joint_target,
                self.q_joint_target,
                cfg.HOLD_DURATION,
                now,
            )

        elif self.current_phase == PHASE_TARGET_HOLD:
            self.print_target_check(q)
            self.start_phase(
                PHASE_RETURN,
                self.q_joint_target,
                self.q_home,
                cfg.MOVE_DURATION,
                now,
            )

        elif self.current_phase == PHASE_RETURN:
            self.start_phase(
                PHASE_RETURN_HOLD,
                self.q_home,
                self.q_home,
                0.1,
                now,
            )

        elif self.current_phase == PHASE_RETURN_HOLD:
            self.print_return_check(q)
            self.task_index += 1

            self.start_phase(
                PHASE_NEXT_WAIT,
                self.q_home,
                self.q_home,
                1.0,
                now,
            )

        elif self.current_phase == PHASE_NEXT_WAIT:
            self.start_joint_task(now)

    def control_loop(self):
        """每个控制周期执行一次。"""
        if self.finished:
            return

        q, dq = self.get_state()
        if q is None or dq is None:
            return

        now = time.monotonic()

        if not self.initialized:
            self.initialize_experiment(q, now)

        q_des, dq_des, ddq_des = self.get_reference(now)

        self.publish_control(
            q,
            dq,
            q_des,
            dq_des,
            ddq_des,
        )

        self.update_phase(q, now)

    def send_emergency_hold(self):
        q, dq = self.get_state()
        if q is None or dq is None:
            return

        self.current_phase = PHASE_EMERGENCY
        q_des, dq_des, ddq_des = hold_reference(q)

        self.publish_control(
            q,
            dq,
            q_des,
            dq_des,
            ddq_des,
        )


def main():
    rclpy.init()
    node = SingleJointMove()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        print()
        print("收到 Ctrl+C，中止运动")

        if rclpy.ok():
            node.send_emergency_hold()

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
