import mujoco
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from mujoco.viewer import launch_passive


class PID:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output
    
def rotate_vector(v, axis, angle_rad):
    axis = axis / np.linalg.norm(axis)
    rot = R.from_rotvec(axis * angle_rad)
    return rot.apply(v)


def simulate_actuator_rotation():
    model = mujoco.MjModel.from_xml_path("roller/scene.xml")
    data = mujoco.MjData(model)

    ref_body_id = model.body("refrence").id
    if ref_body_id == -1:
        raise RuntimeError("Body 'refrence' not found")

    dt = model.opt.timestep

    pid_full = PID(Kp=20.0, Ki=10, Kd=1, dt=dt)

    desired_full_vel = 1.0
    u_a = 1.0
    u_b = 1.0
    r = 0.1
    m = 1.0
    I = 0.1

    # Three bugs: each has actuators and a site for position
    bugs = {
        "rolly1": {
            "actuators": [model.actuator("rolly1_L").id, model.actuator("rolly1_R").id],
            "site": model.site("rolly1_site").id
        },
        "rolly2": {
            "actuators": [model.actuator("rolly2_L").id, model.actuator("rolly2_R").id],
            "site": model.site("rolly2_site").id
        },
        "rolly3": {
            "actuators": [model.actuator("rolly3_L").id, model.actuator("rolly3_R").id],
            "site": model.site("rolly3_site").id
        },
    }

    # Each bug has its own target speed variable
    bug_primes = {name: 0.0 for name in bugs}

    with launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = ref_body_id
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -10

        while viewer.is_running():
            mujoco.mj_step(model, data)

            full_vel = data.sensor("total_vel").data
            full_ang = data.sensor("total_ang").data
            grav_vec = data.sensor("total_grav").data
            grav_vec /= np.linalg.norm(grav_vec)
            
            # Rotate gravity vector 10 degrees CCW around Z-axis
            # grav_vec = rotate_vector(grav_vec, np.array([0, 1, 0]), np.deg2rad(10))

            # Forward and rotational velocities
            x_dot = np.linalg.norm(full_vel)
            theta_dot = full_ang[1]

            # Reference position
            ref_pos = data.xpos[ref_body_id]

            # Loop over each bug
            # --- Classify bugs: 1 B bug (with gravity), 2 A bugs (against gravity) ---
            alignments = {}
            for bug_name, bug_info in bugs.items():
                accel_data = data.sensor(f"{bug_name}_accel").data
                alignment_percent = (accel_data[2] / 9.81) * 100.0
                alignments[bug_name] = alignment_percent
                # print(bug_name, alignment_percent)

            # Bug with min alignment → A (most against gravity)
            bug_a = max(alignments, key=alignments.get)
            # Remaining → B
            bug_b_list = [name for name in bugs if name != bug_a]
            # --- Assign speeds ---
            for bug_name, bug_info in bugs.items():
                L_id, R_id = bug_info["actuators"]
                err = desired_full_vel - x_dot

                if bug_name in bug_b_list:
                    err = desired_full_vel - x_dot
                    bug_primes[bug_name] += pid_full.update(err) * dt
                    force_term = u_b * (bug_primes[bug_name] - x_dot - r * theta_dot)
                else:
                    # alignment <= 0 → against gravity → this bug acts as "a"
                    err = desired_full_vel - x_dot
                    bug_primes[bug_name] += pid_full.update(err) * dt
                    force_term = u_a * (bug_primes[bug_name] - x_dot - r * theta_dot)

                # Apply torque effect to motion (optional integration)
                x_ddot = force_term / m
                theta_ddot = -r * force_term / I
                x_dot = x_ddot * dt
                theta_dot += theta_ddot * dt

                # Apply wheel commands
                L_id, R_id = bug_info["actuators"]
                data.ctrl[L_id] = bug_primes[bug_name]
                data.ctrl[R_id] = -bug_primes[bug_name]

                # print(f"{bug_name} → role: {role}, prime: {bug_primes[bug_name]:.3f}")
            print(f"System: x_dot={x_dot:.3f}, theta_dot={theta_dot:.3f}\n")

            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    simulate_actuator_rotation()