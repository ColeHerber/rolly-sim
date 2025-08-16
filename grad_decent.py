import time
import threading
import os
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

# Headless matplotlib (no macOS windows)
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

# ============================
# User-tweakable parameters
# ============================
RUN_DURATION_S = 30.0      # <-- run for 30 seconds, then report

# Targets
VEL_SETPOINT = 0.75       # m/s
THETA_SETPOINT = -2      # rad/s

# Starting PID gains (edit these)
LIN_START = dict(kp=20.0, ki=3.0, kd=0.5)   # linear speed PID (modifies vel_percent)
ROT_START = dict(kp=20, ki=2.0, kd=0.5)   # rotational speed PID (modifies grav_percent)

# PID output clamps (per-step adjustment bounds)
PID_OUT_MIN, PID_OUT_MAX = -50.0, 50.0
PID_I_LIMIT = 10.0

# Percent clamps (actuator scale factors)
VEL_PCT_MIN, VEL_PCT_MAX = -100.0, 100.0
GRAV_PCT_MIN, GRAV_PCT_MAX = -100.0, 100.0

# Adaptation hyperparameters
ADAPT_WINDOW_S = 4.0       # seconds per measurement window
FD_DELTA       = 0.5       # finite-difference perturbation for a single gain
GD_LR          = 0.25      # gradient descent learning rate
COST_W_LIN     = 1.0       # weight on linear speed squared error
COST_W_ROT     = 1.0       # weight on angular speed squared error

# Snapshot/plot settings
SS_WINDOW_S = 1.0          # rolling std window (only for annotating snapshot)
SS_LOOKBACK = 5.0          # lookback (only for annotating snapshot)
SS_RATIO    = 0.05         # std ratio (only for annotating snapshot)

# Headless run pacing
SLOWDOWN_SLEEP = 0.0       # fastest headless

# ============================
# PID Controller
# ============================
class PID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, out_min=None, out_max=None, integral_limit=None):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.out_min = out_min; self.out_max = out_max
        self.integral_limit = integral_limit
        self._i = 0.0; self._prev_e = None

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
        if self.out_min is not None: u = max(self.out_min, u)
        if self.out_max is not None: u = min(self.out_max, u)
        self._prev_e = error
        return u

# ============================
# Steady-state helpers (for snapshot annotation only)
# ============================
def _rolling_std(t, y, window=1.0):
    y = np.asarray(y, dtype=float); t = np.asarray(t, dtype=float)
    stds = np.zeros_like(y)
    j = 0
    for i in range(len(y)):
        while j < i and (t[i] - t[j]) > window:
            j += 1
        seg = y[j:i+1]
        stds[i] = np.std(seg) if seg.size >= 2 else 0.0
    return stds

def find_steady_index(t, lin, ang, window=1.0, lookback=5.0, ratio=0.05):
    t = np.asarray(t, dtype=float); lin = np.asarray(lin, dtype=float); ang = np.asarray(ang, dtype=float)
    if t.size < 3: return None
    std_lin = _rolling_std(t, lin, window)
    std_ang = _rolling_std(t, ang, window)
    for k in range(len(t)):
        tt = t[k] - lookback
        if tt < t[0]: continue
        j = np.searchsorted(t, tt)
        if j >= len(t): j = len(t) - 1
        if j == k and j > 0: j -= 1
        if std_lin[j] <= 0.0 and std_ang[j] <= 0.0: continue
        ok_lin = std_lin[k] <= ratio * std_lin[j] if std_lin[j] > 0 else False
        ok_ang = std_ang[k] <= ratio * std_ang[j] if std_ang[j] > 0 else False
        if ok_lin and ok_ang: return k
    return None

# ============================
# Snapshot plot (full history)
# ============================
def save_snapshot(time_hist, xdot_hist, thetadot_hist, vel_setpoint, theta_setpoint):
    if not time_hist:
        print("No data yet—try again shortly."); return
    os.makedirs("figures", exist_ok=True)
    t = np.array(time_hist, dtype=float)
    lin = np.array(xdot_hist, dtype=float)
    ang = np.array(thetadot_hist, dtype=float)

    k_ss = find_steady_index(t, lin, ang, SS_WINDOW_S, SS_LOOKBACK, SS_RATIO)
    title = ("Steady state NOT detected" if k_ss is None else f"Steady state at t = {t[k_ss]:.2f}s")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, lin, label="Linear speed |v| (m/s)")
    axes[0].hlines(vel_setpoint, t[0], t[-1], linestyles="--", linewidth=1, label="Linear target")
    axes[0].set_ylabel("m/s"); axes[0].grid(True); axes[0].legend(loc="upper right")

    axes[1].plot(t, ang, label="Angular speed ω_y (rad/s)")
    axes[1].hlines(theta_setpoint, t[0], t[-1], linestyles="--", linewidth=1, label="Angular target")
    axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("rad/s"); axes[1].grid(True); axes[1].legend(loc="upper right")

    if k_ss is not None:
        for ax in axes: ax.axvline(t[k_ss], linestyle="--", linewidth=1)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = f"figures/steady_snapshot_{ts}.png"
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Saved snapshot → {out_path}")

# ============================
# Adaptation state machine
# ============================
class PIDTuner:
    """
    Coordinate-wise finite-difference gradient descent:
    - Measure cost over a window with current gains (BASE)
    - Perturb one gain by +delta for the next window (PERTURB)
    - Estimate dJ/dtheta and update with learning rate
    Gains are constrained to be >= 0.
    """
    def __init__(self, lin_pid: PID, rot_pid: PID, lr=0.25, delta=0.5):
        self.lin_pid = lin_pid
        self.rot_pid = rot_pid
        self.lr = lr
        self.delta = delta
        self.params = [('lin','kp'), ('lin','ki'), ('lin','kd'),
                       ('rot','kp'), ('rot','ki'), ('rot','kd')]
        self.idx = 0
        self.phase = 'BASE'   # 'BASE' -> 'PERTURB'
        self.base_cost = None
        self.param_backup = None

    def _get_param_ref(self, which, name):
        pid = self.lin_pid if which=='lin' else self.rot_pid
        return pid, name

    def begin_window(self):
        which, name = self.params[self.idx]
        pid, attr = self._get_param_ref(which, name)
        if self.phase == 'BASE':
            self.param_backup = getattr(pid, attr)
        elif self.phase == 'PERTURB':
            self.param_backup = getattr(pid, attr)
            setattr(pid, attr, max(0.0, self.param_backup + self.delta))
        pid.reset()

    def end_window_and_update(self, cost_value):
        which, name = self.params[self.idx]
        pid, attr = self._get_param_ref(which, name)
        if self.phase == 'BASE':
            self.base_cost = cost_value
            self.phase = 'PERTURB'
        else:
            J_base = self.base_cost; J_pert = cost_value
            grad = (J_pert - J_base) / max(1e-9, self.delta)
            new_val = self.param_backup - self.lr * grad  # gradient descent
            setattr(pid, attr, max(0.0, new_val))
            pid.reset()
            self.idx = (self.idx + 1) % len(self.params)
            self.phase = 'BASE'
            self.base_cost = None
            self.param_backup = None

    def current_param_name(self):
        which, name = self.params[self.idx]
        return f"{which}.{name} ({self.phase})"

# ============================
# Main simulation (headless, 30s)
# ============================
def simulate_headless_30s():
    # Load model/data
    model = mujoco.MjModel.from_xml_path("roller/scene.xml")
    data = mujoco.MjData(model)

    # Actuators & sensors
    bugs = {
        "rolly1": [model.actuator("rolly1_L").id, model.actuator("rolly1_R").id, "rolly1_accel", "rolly1_gyro"],
        "rolly2": [model.actuator("rolly2_L").id, model.actuator("rolly2_R").id, "rolly2_accel", "rolly2_gyro"],
        "rolly3": [model.actuator("rolly3_L").id, model.actuator("rolly3_R").id, "rolly3_accel", "rolly3_gyro"],
    }

    # Gravity vector (optionally rotated)
    grav = np.array([0.0, -9.81, 0.0])
    axis = np.array([1.0, 0.0, 0.0])
    angle_degrees = 0.0
    grav = R.from_rotvec(np.radians(angle_degrees) * axis).apply(grav)
    gnorm = np.linalg.norm(grav)

    # PIDs (starting gains)
    lin_pid = PID(**LIN_START, out_min=PID_OUT_MIN, out_max=PID_OUT_MAX, integral_limit=PID_I_LIMIT)
    rot_pid = PID(**ROT_START, out_min=PID_OUT_MIN, out_max=PID_OUT_MAX, integral_limit=PID_I_LIMIT)
    tuner = PIDTuner(lin_pid, rot_pid, lr=GD_LR, delta=FD_DELTA)

    # Histories (for plots and error stats)
    time_hist, xdot_hist, thetadot_hist = [], [], []

    # Percent scalers
    vel_percent  = 20.0
    grav_percent = 6.0

    prev_time = data.time
    window_start_time = data.time
    sim_start_time = data.time
    tuner.begin_window()

    # Snapshot trigger via Enter (does NOT stop sim)
    plot_event = threading.Event()
    def plot_on_enter():
        while True:
            try:
                input("Press Enter to save a snapshot (simulation keeps running)...\n")
            except EOFError:
                break
            plot_event.set()
    threading.Thread(target=plot_on_enter, daemon=True).start()

    # Running cost accumulation for adaptation window
    cost_sum = 0.0
    samples  = 0

    print("Running headless for 30 seconds… Press Enter to save snapshots; Ctrl+C to abort early.")
    try:
        while data.time - sim_start_time < RUN_DURATION_S:
            # Sensors
            full_vel = data.sensor("total_vel").data
            full_ang = data.sensor("total_ang").data
            x_dot = float(np.linalg.norm(full_vel))
            theta_dot = float(full_ang[1])

            # Timing
            dt = data.time - prev_time
            if dt <= 0.0: dt = model.opt.timestep
            prev_time = data.time

            # Errors
            vel_error  = VEL_SETPOINT   - x_dot
            rot_error  = THETA_SETPOINT - theta_dot

            # Outer-loop PIDs (update scalers)
            vel_percent  += lin_pid.update(vel_error, dt)
            grav_percent += rot_pid.update(rot_error, dt)
            vel_percent   = float(np.clip(vel_percent,  VEL_PCT_MIN,  VEL_PCT_MAX))
            grav_percent  = float(np.clip(grav_percent, GRAV_PCT_MIN, GRAV_PCT_MAX))

            # Alignment per robot
            align_map = {}
            if gnorm > 1e-9:
                for name, (_L, _R, accel_name, _gyro_name) in bugs.items():
                    accel = data.sensor(accel_name).data
                    anorm = np.linalg.norm(accel)
                    align_map[name] = float(np.dot(accel, grav) / (anorm * gnorm)) if anorm > 1e-9 else 0.0
            else:
                for name in bugs.keys(): align_map[name] = 0.0

            # Forward component of global velocity (projection onto [0, -1, 0])
            forward_component = float(np.dot(full_vel, [0.0, -1.0, 0.0]))

            # Apply commands per robot
            for name, (L_id, R_id, _accel_name, _gyro_name) in bugs.items():
                speed_grav = grav_percent * align_map[name]
                speed_vel  = vel_percent  * forward_component
                speed_cmd  = -speed_grav + speed_vel
                data.ctrl[L_id] = np.float64(speed_cmd)
                data.ctrl[R_id] = np.float64(-speed_cmd)

            # Log histories
            time_hist.append(data.time)
            xdot_hist.append(x_dot)
            thetadot_hist.append(theta_dot)

            # Accumulate cost for adaptation window
            cost_sum += (COST_W_LIN * vel_error * vel_error + COST_W_ROT * rot_error * rot_error)
            samples  += 1

            # Step sim
            mujoco.mj_step(model, data)

            # Adaptation window boundary
            if (data.time - window_start_time) >= ADAPT_WINDOW_S:
                avg_cost = cost_sum / max(1, samples)
                tuner.end_window_and_update(avg_cost)
                # Optional: comment next line for quieter output
                print(f"[t={data.time:6.2f}s] cost={avg_cost:.6f}, tuning {tuner.current_param_name()}")
                cost_sum = 0.0; samples = 0
                window_start_time = data.time
                tuner.begin_window()

            # Enter-triggered snapshot (full history)
            if plot_event.is_set():
                save_snapshot(time_hist, xdot_hist, thetadot_hist, VEL_SETPOINT, THETA_SETPOINT)
                plot_event.clear()

            # Optional pacing (kept 0.0 for max speed)
            if SLOWDOWN_SLEEP > 0.0:
                time.sleep(SLOWDOWN_SLEEP)

    except KeyboardInterrupt:
        print("\nAborted early by user.")

    # ===== Final report after 30s (or abort) =====
    t_now = time_hist[-1] if time_hist else 0.0
    t0 = t_now - 5.0
    j0 = np.searchsorted(time_hist, t0) if time_hist else 0
    lin_err_mae = float(np.mean(np.abs(VEL_SETPOINT - np.array(xdot_hist[j0:], dtype=float)))) if j0 < len(xdot_hist) else float('nan')
    rot_err_mae = float(np.mean(np.abs(THETA_SETPOINT - np.array(thetadot_hist[j0:], dtype=float)))) if j0 < len(thetadot_hist) else float('nan')

    print("\n=== Run complete (30s) ===")
    print(f"Final Linear PID:     kp={lin_pid.kp:.4f}, ki={lin_pid.ki:.4f}, kd={lin_pid.kd:.4f}")
    print(f"Final Rotational PID: kp={rot_pid.kp:.4f}, ki={rot_pid.ki:.4f}, kd={rot_pid.kd:.4f}")
    print(f"Avg |error| over last 5.0s: linear={lin_err_mae:.5f} m/s, angular={rot_err_mae:.5f} rad/s")

if __name__ == "__main__":
    simulate_headless_30s()