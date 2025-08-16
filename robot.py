import time
import threading
import os
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from mujoco import viewer

# Headless matplotlib to avoid macOS NSWindow-on-non-main-thread
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# ----------------------------
# Simple PID Controller
# ----------------------------
class PID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, out_min=None, out_max=None, integral_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral_limit = integral_limit
        self._i = 0.0
        self._prev_e = None

    def reset(self):
        self._i = 0.0
        self._prev_e = None

    def update(self, error, dt):
        d = 0.0 if (dt <= 0.0 or self._prev_e is None) else (error - self._prev_e) / dt
        self._i += error * dt
        if self.integral_limit is not None:
            lim = abs(self.integral_limit)
            self._i = max(-lim, min(lim, self._i))
        u = self.kp * error + self.ki * self._i + self.kd * d
        if self.out_min is not None:
            u = max(self.out_min, u)
        if self.out_max is not None:
            u = min(self.out_max, u)
        self._prev_e = error
        return u

# ----------------------------
# Steady-state detection utils
# ----------------------------
def _rolling_std(t, y, window=1.0):
    """Rolling std over a trailing time window (seconds)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    stds = np.zeros_like(y)
    j = 0
    for i in range(len(y)):
        while j < i and (t[i] - t[j]) > window:
            j += 1
        seg = y[j:i+1]
        stds[i] = np.std(seg) if seg.size >= 2 else 0.0
    return stds

def _find_steady_time(t, lin, ang, window=1.0, lookback=5.0, ratio=0.05):
    """
    Return index k of first time where rolling std_now <= ratio * std_prev
    for BOTH linear and angular speed, with prev measured lookback seconds earlier.
    If not found, return None.
    """
    t = np.asarray(t, dtype=float)
    lin = np.asarray(lin, dtype=float)
    ang = np.asarray(ang, dtype=float)
    if t.size < 3:
        return None

    std_lin = _rolling_std(t, lin, window)
    std_ang = _rolling_std(t, ang, window)

    for k in range(len(t)):
        target_time = t[k] - lookback
        if target_time < t[0]:
            continue
        j = np.searchsorted(t, target_time)
        if j >= len(t):
            j = len(t) - 1
        if j == k and j > 0:
            j -= 1

        if std_lin[j] <= 0.0 and std_ang[j] <= 0.0:
            continue

        cond_lin = std_lin[k] <= ratio * std_lin[j] if std_lin[j] > 0 else False
        cond_ang = std_ang[k] <= ratio * std_ang[j] if std_ang[j] > 0 else False

        if cond_lin and cond_ang:
            return k
    return None

def save_snapshot(time_hist, xdot_hist, thetadot_hist, vel_setpoint, theta_setpoint):
    """Save a plot of |v| and ω_y (with target lines) over the full time history."""
    if not time_hist:
        print("No data yet—try again shortly.")
        return

    os.makedirs("figures", exist_ok=True)

    t = np.array(time_hist, dtype=float)
    lin = np.array(xdot_hist, dtype=float)
    ang = np.array(thetadot_hist, dtype=float)

    # Detect steady-state (still reported, but we won’t truncate)
    k_ss = _find_steady_time(t, lin, ang, window=1.0, lookback=5.0, ratio=0.05)
    reached_msg = ("Steady state NOT detected"
                   if k_ss is None else f"Steady state at t = {t[k_ss]:.2f}s")

    # Build figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Linear speed with target
    axes[0].plot(t, lin, label="Linear speed |v| (m/s)")
    axes[0].hlines(vel_setpoint, t[0], t[-1], linestyles="--", linewidth=1, label="Linear target")
    axes[0].set_ylabel("m/s")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    # Angular speed with target
    axes[1].plot(t, ang, label="Angular speed ω_y (rad/s)")
    axes[1].hlines(theta_setpoint, t[0], t[-1], linestyles="--", linewidth=1, label="Angular target")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("rad/s")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    # Mark steady-state time if detected
    if k_ss is not None:
        for ax in axes:
            ax.axvline(t[k_ss], linestyle="--", linewidth=1)

    fig.suptitle(reached_msg)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = f"figures/steady_snapshot_{ts}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved snapshot → {out_path}")
# ----------------------------
# Main simulation
# ----------------------------
def simulate_and_watch():
    """
    Open interactive viewer, run until the window is closed.
    Press Enter in the terminal to SAVE a steady-state snapshot (simulation keeps running).
    Rotation PID modifies grav_percent; velocity PID modifies vel_percent.
    """
    # --- Load model & data ---
    model = mujoco.MjModel.from_xml_path("roller/scene.xml")
    data = mujoco.MjData(model)

    # --- Actuators & sensors ---
    # [L_id, R_id, accel_sensor_name, gyro_sensor_name]
    bugs = {
        "rolly1": [model.actuator("rolly1_L").id, model.actuator("rolly1_R").id, "rolly1_accel", "rolly1_gyro"],
        "rolly2": [model.actuator("rolly2_L").id, model.actuator("rolly2_R").id, "rolly2_accel", "rolly2_gyro"],
        "rolly3": [model.actuator("rolly3_L").id, model.actuator("rolly3_R").id, "rolly3_accel", "rolly3_gyro"],
    }

    # --- Gravity vector (optionally rotated) ---
    grav = np.array([0.0, -9.81, 0.0])
    axis = np.array([1.0, 0.0, 0.0])
    angle_degrees = 0.0
    grav = R.from_rotvec(np.radians(angle_degrees) * axis).apply(grav)
    gnorm = np.linalg.norm(grav)

    # ----------------------------
    # Control targets & scalers
    # ----------------------------
    vel_setpoint = 0.3       # desired speed magnitude [m/s]
    theta_setpoint = -6    # desired rotational rate about Y [rad/s] (negative target supported)

    grav_percent = 6.0       # rotation scaler (PID will adjust)
    vel_percent  = 20.0      # linear velocity scaler (PID will adjust)

    # Clamp ranges per request
    GRAV_PCT_MIN, GRAV_PCT_MAX = -100.0, 100.0
    VEL_PCT_MIN,  VEL_PCT_MAX  = -100.0, 100.0

    # PID gains (tune for your scene)
    rot_pid = PID(kp=17.35, ki=8, kd=1.6, out_min=-50.0, out_max=50.0, integral_limit=10.0)  # modifies grav_percent
    lin_pid = PID(kp=10.5, ki=2, kd=0.5, out_min=-50.0, out_max=50.0, integral_limit=10.0)  # modifies vel_percent

    # --- Histories for steady-state snapshot ---
    time_hist = []
    xdot_hist = []
    thetadot_hist = []

    prev_time = data.time

    # Visibility pacing
    SLOWDOWN_SLEEP = 0.001  # seconds

    # --- Snapshot trigger via Enter key (does NOT stop sim) ---
    plot_event = threading.Event()
    def plot_on_enter():
        while True:
            try:
                input("Press Enter to save a steady-state snapshot (simulation keeps running)...\n")
            except EOFError:
                break
            plot_event.set()
    threading.Thread(target=plot_on_enter, daemon=True).start()

    # --- Viewer loop ---
    with viewer.launch_passive(model, data) as v:
        v.cam.distance = max(v.cam.distance, 5.0)

        while v.is_running():
            # Read sensors
            full_vel = data.sensor("total_vel").data    # vec3 linear velocity at reference site
            full_ang = data.sensor("total_ang").data    # vec3 angular rates (rad/s)

            x_dot = float(np.linalg.norm(full_vel))     # speed magnitude
            theta_dot = float(full_ang[1])              # rotational rate about Y

            # Timing
            dt = data.time - prev_time
            if dt <= 0.0:
                dt = model.opt.timestep
            prev_time = data.time

            # Outer-loop PIDs (signed errors handle negative theta_setpoint)
            vel_error = vel_setpoint - x_dot
            rot_error = theta_setpoint - theta_dot

            vel_percent  += lin_pid.update(vel_error, dt)
            grav_percent += rot_pid.update(rot_error, dt)

            vel_percent  = float(np.clip(vel_percent,  VEL_PCT_MIN,  VEL_PCT_MAX))
            grav_percent = float(np.clip(grav_percent, GRAV_PCT_MIN, GRAV_PCT_MAX))

            # Alignment per robot
            align_map = {}
            if gnorm > 1e-9:
                for name, (_L, _R, accel_name, _gyro_name) in bugs.items():
                    accel = data.sensor(accel_name).data
                    anorm = np.linalg.norm(accel)
                    if anorm > 1e-9:
                        align_map[name] = float(np.dot(accel, grav) / (anorm * gnorm))  # [-1, 1]
                    else:
                        align_map[name] = 0.0
            else:
                for name in bugs.keys():
                    align_map[name] = 0.0

            # Forward component of global velocity (original projection onto [0, -1, 0])
            forward_component = float(np.dot(full_vel, [0.0, -1.0, 0.0]))

            # Apply commands per robot
            for name, (L_id, R_id, _accel_name, _gyro_name) in bugs.items():
                speed_grav = grav_percent * align_map[name]
                speed_vel  = vel_percent  * forward_component
                speed_cmd  = -speed_grav + speed_vel

                data.ctrl[L_id] = np.float64(speed_cmd)
                data.ctrl[R_id] = np.float64(-speed_cmd)

            # Log global speeds
            time_hist.append(data.time)
            xdot_hist.append(x_dot)
            thetadot_hist.append(theta_dot)

            # Step and render
            mujoco.mj_step(model, data)
            v.sync()

            # Slow down for visibility
            if SLOWDOWN_SLEEP > 0.0:
                time.sleep(SLOWDOWN_SLEEP)

            # If user pressed Enter, save a steady-state snapshot (no GUI)
            if plot_event.is_set():
                save_snapshot(time_hist, xdot_hist, thetadot_hist, vel_setpoint, theta_setpoint)
                plot_event.clear()

if __name__ == "__main__":
    simulate_and_watch()