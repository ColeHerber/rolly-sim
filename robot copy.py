import mujoco
import numpy as np
import time
from mujoco.viewer import launch_passive
from collections import deque
import matplotlib.pyplot as plt


def plot_sensor_history(torque_log, gyro_log):
    """
    Plot stored torque and gyro histories for each roller on exit.
    Generates combined and separate plots for raw values and changes.
    """
    if not torque_log:
        return

    markers = ['o', 's', '^', 'd', 'v', 'P']
    color_cycle = plt.cm.get_cmap("tab10", max(len(torque_log), len(markers)))

    torque_avg = {}
    torque_delta = {}
    gyro_mag = {}
    gyro_delta = {}

    sample_start = 30
    sample_end = 2500

    def mask_by_threshold(values, factor=0):
        if len(values) == 0:
            return None
        max_val = np.nanmax(values)
        if not np.isfinite(max_val) or max_val <= 0:
            return None
        threshold = factor * max_val
        mask = values >= threshold
        if not np.any(mask):
            return None
        return np.where(mask, values, np.nan)

    for name, samples in torque_log.items():
        if not samples:
            continue
        avg_all = 0.5 * (np.array([left for left, _ in samples]) + np.array([right for _, right in samples]))
        avg_vals = np.clip(avg_all[sample_start:sample_end], 1e-9, None)
        torque_avg[name] = avg_vals
        if len(avg_vals) > 1:
            alpha_torque = 0.05
            smooth = np.empty_like(avg_vals)
            smooth[0] = avg_vals[0]
            for i in range(1, len(avg_vals)):
                smooth[i] = alpha_torque * avg_vals[i] + (1 - alpha_torque) * smooth[i - 1]
            high_pass = np.clip(np.abs(avg_vals - smooth), 1e-9, None)
        else:
            high_pass = np.zeros_like(avg_vals)
        torque_delta[name] = high_pass

    for name, samples in gyro_log.items():
        if not samples:
            continue
        comp = np.array(samples, dtype=float)
        mag_all = np.linalg.norm(comp, axis=1)
        mag_vals = np.clip(mag_all[sample_start:sample_end], 1e-9, None)
        gyro_mag[name] = mag_vals
        # High-pass filter: aggressive smoothing (low-pass) then subtract
        if len(mag_vals) > 1:
            alpha = 0.05
            smooth = np.empty_like(mag_vals)
            smooth[0] = mag_vals[0]
            for i in range(1, len(mag_vals)):
                smooth[i] = alpha * mag_vals[i] + (1 - alpha) * smooth[i - 1]
            high_pass = np.clip(np.abs(mag_vals - smooth), 1e-9, None)
        else:
            high_pass = np.zeros_like(mag_vals)
        gyro_delta[name] = high_pass

    # Combined torque + gyro plot with dual y-axes
    fig, ax_left = plt.subplots(figsize=(12, 7))
    ax_right = ax_left.twinx()
    left_handles = []
    right_handles = []
    for idx, name in enumerate(sorted(torque_log.keys())):
        color = color_cycle(idx % color_cycle.N)
        marker = markers[idx % len(markers)]
        if name in gyro_mag:
            x_vals = np.arange(sample_start, sample_start + len(gyro_mag[name]))
            line_left, = ax_left.plot(
                x_vals,
                gyro_mag[name],
                label=f"{name} gyro",
                color=color,
                linewidth=1.6,
            )
            left_handles.append(line_left)
        if name in torque_avg:
            x_vals = np.arange(sample_start, sample_start + len(torque_avg[name]))
            line_right, = ax_right.plot(
                x_vals,
                torque_avg[name],
                label=f"{name} torque",
                color=color,
                linestyle="--",
                marker=marker,
                markevery=[len(torque_avg[name]) - 1],
                linewidth=1.5,
            )
            right_handles.append(line_right)
    ax_left.set_title("Roller Gyro & Torque History")
    ax_left.set_xlabel("Sample")
    ax_left.set_ylabel("Gyro Magnitude (log scale)")
    ax_right.set_ylabel("Torque (log scale)")
    ax_left.set_yscale("log")
    ax_right.set_yscale("log")
    ax_left.grid(True, linestyle="--", alpha=0.3)
    handles = left_handles + right_handles
    labels = [h.get_label() for h in handles]
    ax_left.legend(handles, labels, loc="lower left")
    fig.tight_layout()
    plt.savefig("torque_gyro_curve")
    

    # Combined change plot with dual y-axes
    fig_delta, ax_delta_left = plt.subplots(figsize=(12, 7))
    ax_delta_right = ax_delta_left.twinx()
    left_handles = []
    right_handles = []
    for idx, name in enumerate(sorted(torque_log.keys())):
        color = color_cycle(idx % color_cycle.N)
        if name in gyro_delta:
            masked_gyro = mask_by_threshold(gyro_delta[name])
            if masked_gyro is not None:
                x_vals = np.arange(sample_start, sample_start + len(masked_gyro))
                line_left, = ax_delta_left.plot(
                    x_vals,
                    masked_gyro,
                    label=f"{name} gyro Δ",
                    color=color,
                    linewidth=1.4,
                )
                left_handles.append(line_left)
        if name in torque_delta:
            masked_torque = mask_by_threshold(torque_delta[name])
            if masked_torque is not None:
                x_vals = np.arange(sample_start, sample_start + len(masked_torque))
                line_right, = ax_delta_right.plot(
                    x_vals,
                    masked_torque,
                    label=f"{name} torque Δ",
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                )
                right_handles.append(line_right)
    ax_delta_left.set_title("Change in Gyro & Torque Readings")
    ax_delta_left.set_xlabel("Sample")
    ax_delta_left.set_ylabel("Gyro Change (log scale)")
    ax_delta_right.set_ylabel("Torque Change (log scale)")
    ax_delta_left.set_yscale("log")
    ax_delta_right.set_yscale("log")
    ax_delta_left.grid(True, linestyle="--", alpha=0.3)
    handles = left_handles + right_handles
    labels = [h.get_label() for h in handles]
    ax_delta_left.legend(handles, labels, loc="lower left")
    fig_delta.tight_layout()
    plt.savefig("torque_gyro_change")
    

    # Stacked torque & gyro history plots
    fig_stacked_hist, (ax_gyro_hist, ax_torque_hist) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for idx, name in enumerate(sorted(torque_log.keys())):
        color = color_cycle(idx % color_cycle.N)
        marker = markers[idx % len(markers)]
        if name in gyro_mag:
            x_vals = np.arange(sample_start, sample_start + len(gyro_mag[name]))
            ax_gyro_hist.plot(
                x_vals,
                gyro_mag[name],
                label=f"{name} gyro",
                color=color,
                linewidth=1.6,
            )
        if name in torque_avg:
            x_vals = np.arange(sample_start, sample_start + len(torque_avg[name]))
            ax_torque_hist.plot(
                x_vals,
                torque_avg[name],
                label=f"{name} torque",
                color=color,
                linestyle="--",
                marker=marker,
                markevery=[len(torque_avg[name]) - 1],
                linewidth=1.5,
            )
    ax_gyro_hist.set_title("Gyro Magnitude (Log Scale)")
    ax_gyro_hist.set_ylabel("Gyro Magnitude")
    ax_gyro_hist.set_yscale("log")
    ax_gyro_hist.grid(True, linestyle="--", alpha=0.3)
    ax_gyro_hist.legend(loc="lower left")

    ax_torque_hist.set_title("Torque (Log Scale)")
    ax_torque_hist.set_xlabel("Sample")
    ax_torque_hist.set_ylabel("Torque")
    ax_torque_hist.set_yscale("log")
    ax_torque_hist.grid(True, linestyle="--", alpha=0.3)
    ax_torque_hist.legend(loc="lower left")

    fig_stacked_hist.tight_layout()
    plt.savefig("torque_gyro_stacked_curve")
    

    # Torque-only plot
    fig_torque, ax_torque = plt.subplots(figsize=(12, 7))
    for idx, name in enumerate(sorted(torque_avg.keys())):
        marker = markers[idx % len(markers)]
        color = color_cycle(idx % color_cycle.N)
        x_vals = np.arange(sample_start, sample_start + len(torque_avg[name]))
        ax_torque.plot(
            x_vals,
            torque_avg[name],
            label=name,
            color=color,
            marker=marker,
            markevery=[len(torque_avg[name]) - 1],
            linewidth=1.75,
        )
    ax_torque.set_title("Roller Torque History (Average Left/Right)")
    ax_torque.set_xlabel("Sample")
    ax_torque.set_ylabel("Torque (log scale)")
    ax_torque.set_yscale("log")
    ax_torque.legend(loc="lower left")
    ax_torque.grid(True, linestyle="--", alpha=0.3)
    fig_torque.tight_layout()
    plt.savefig("torque_curve_only")
    

    # Torque change plot
    fig_torque_delta, ax_torque_delta = plt.subplots(figsize=(12, 7))
    for idx, name in enumerate(sorted(torque_delta.keys())):
        color = color_cycle(idx % color_cycle.N)
        masked = mask_by_threshold(torque_delta[name])
        if masked is not None:
            x_vals = np.arange(sample_start, sample_start + len(masked))
            ax_torque_delta.plot(
                x_vals,
                masked,
                label=name,
                color=color,
                linewidth=1.5,
            )
    ax_torque_delta.set_title("Change in Roller Torque (High-Pass)")
    ax_torque_delta.set_xlabel("Sample")
    ax_torque_delta.set_ylabel("Torque Change (log scale)")
    ax_torque_delta.set_yscale("log")
    ax_torque_delta.legend(loc="lower left")
    ax_torque_delta.grid(True, linestyle="--", alpha=0.3)
    fig_torque_delta.tight_layout()
    plt.savefig("torque_change_only")
    

    # Stacked torque & gyro change plots
    fig_stacked_delta, (ax_gyro_delta_top, ax_torque_delta_bottom) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for idx, name in enumerate(sorted(torque_log.keys())):
        color = color_cycle(idx % color_cycle.N)
        if name in gyro_delta:
            masked_gyro = mask_by_threshold(gyro_delta[name])
            if masked_gyro is not None:
                x_vals = np.arange(sample_start, sample_start + len(masked_gyro))
                ax_gyro_delta_top.plot(
                    x_vals,
                    masked_gyro,
                    label=f"{name} gyro Δ",
                    color=color,
                    linewidth=1.5,
                )
        if name in torque_delta:
            masked_torque = mask_by_threshold(torque_delta[name])
            if masked_torque is not None:
                x_vals = np.arange(sample_start, sample_start + len(masked_torque))
                ax_torque_delta_bottom.plot(
                    x_vals,
                    masked_torque,
                    label=f"{name} torque Δ",
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                )
    ax_gyro_delta_top.set_title("Gyro Change (High-Pass, Log Scale)")
    ax_gyro_delta_top.set_ylabel("Gyro Change")
    ax_gyro_delta_top.set_yscale("log")
    ax_gyro_delta_top.grid(True, linestyle="--", alpha=0.3)
    ax_gyro_delta_top.legend(loc="lower left")

    ax_torque_delta_bottom.set_title("Torque Change (High-Pass, Log Scale)")
    ax_torque_delta_bottom.set_xlabel("Sample")
    ax_torque_delta_bottom.set_ylabel("Torque Change")
    ax_torque_delta_bottom.set_yscale("log")
    ax_torque_delta_bottom.grid(True, linestyle="--", alpha=0.3)
    ax_torque_delta_bottom.legend(loc="lower left")

    fig_stacked_delta.tight_layout()
    plt.savefig("torque_gyro_stacked_change")
    

    # Gyro-only plot
    fig_gyro, ax_gyro = plt.subplots(figsize=(12, 7))
    for idx, name in enumerate(sorted(gyro_mag.keys())):
        marker = markers[idx % len(markers)]
        color = color_cycle(idx % color_cycle.N)
        x_vals = np.arange(sample_start, sample_start + len(gyro_mag[name]))
        ax_gyro.plot(
            x_vals,
            gyro_mag[name],
            label=name,
            color=color,
            marker=marker,
            markevery=[len(gyro_mag[name]) - 1],
            linewidth=1.75,
        )
    ax_gyro.set_title("Roller Gyro History (Magnitude)")
    ax_gyro.set_xlabel("Sample")
    ax_gyro.set_ylabel("Gyro Magnitude (log scale)")
    ax_gyro.set_yscale("log")
    ax_gyro.legend(loc="lower left")
    ax_gyro.grid(True, linestyle="--", alpha=0.3)
    fig_gyro.tight_layout()
    plt.savefig("gyro_curve_only")
    

    # Gyro change plot
    fig_gyro_delta, ax_gyro_delta = plt.subplots(figsize=(12, 7))
    for idx, name in enumerate(sorted(gyro_delta.keys())):
        color = color_cycle(idx % color_cycle.N)
        masked = mask_by_threshold(gyro_delta[name])
        if masked is not None:
            x_vals = np.arange(sample_start, sample_start + len(masked))
            ax_gyro_delta.plot(
                x_vals,
                masked,
                label=name,
                color=color,
                linewidth=1.5,
            )
    ax_gyro_delta.set_title("Change in Roller Gyro Magnitude")
    ax_gyro_delta.set_xlabel("Sample")
    ax_gyro_delta.set_ylabel("Gyro Change (log scale)")
    ax_gyro_delta.set_yscale("log")
    ax_gyro_delta.legend(loc="lower left")
    ax_gyro_delta.grid(True, linestyle="--", alpha=0.3)
    fig_gyro_delta.tight_layout()
    plt.savefig("gyro_change_only")
    


def roller_actuator_rotation():
    """
    Loads a MuJoCo model from XML, sets target velocities for actuators,
    and runs a roller with visualization.
    """
    try:
        # Load the MuJoCo model from the XML string
        model = mujoco.MjModel.from_xml_path("roller/scene.xml")
        data = mujoco.MjData(model)

        # Ensure the model and data are valid
        if model is None:
            print("Error: Could not load MuJoCo model from XML string.")
            return
        
        target_body_name = 'control'
        target_body_id = model.body(target_body_name).id


        if target_body_id == -1:
            print(f"Error: Body '{target_body_name}' not found in the model.")
            exit()

        command_history_length = 100
        speeds = np.concatenate((np.ones((command_history_length//2))*5, np.ones((command_history_length//2))*5))
        measured_torques = np.zeros(command_history_length)

        bugs = {
            "roller1": [model.actuator("L1").id, model.actuator("R1").id, model.site("roller1").id, "roller1_accel", "roller1_gyro", "L1_w", "R1_w", "0.5", "L1_torque", "R1_torque", speeds, measured_torques],
            "roller2": [model.actuator("L2").id, model.actuator("R2").id, model.site("roller2").id, "roller2_accel", "roller2_gyro", "L2_w", "R2_w", "0.5", "L2_torque", "R2_torque", speeds, measured_torques],
            "roller3": [model.actuator("L3").id, model.actuator("R3").id, model.site("roller3").id, "roller3_accel", "roller3_gyro", "L3_w", "R3_w", "0.5", "L3_torque", "R3_torque", speeds, measured_torques],
            "roller4": [model.actuator("L4").id, model.actuator("R4").id, model.site("roller4").id, "roller4_accel", "roller4_gyro", "L4_w", "R4_w", "0.5", "L4_torque", "R4_torque", speeds, measured_torques],
            "roller5": [model.actuator("L5").id, model.actuator("R5").id, model.site("roller5").id, "roller5_accel", "roller5_gyro", "L5_w", "R5_w", "0.5", "L5_torque", "R5_torque", speeds, measured_torques],
            "roller6": [model.actuator("L6").id, model.actuator("R6").id, model.site("roller6").id, "roller6_accel", "roller6_gyro", "L6_w", "R6_w", "0.5", "L6_torque", "R6_torque", speeds, measured_torques],

        }
        torque_log = {name: [] for name in bugs}
        gyro_log = {name: [] for name in bugs}

        grav = np.array([0,0,1])


        labels = [
            "ax", "ay", "az", 
            "vx", "vy", "vz", 
             "grav%", "command", "L_w", "R_w", "L_tau", "R_tau"
                ]
        label_line = " ".join(f"{label:>7}" for label in labels)



        do_print = False
        current_pos = 0
        # Initialize the viewer
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:

                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = target_body_id

                        # Set the camera to tracking mode
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = target_body_id

                # Optional: Adjust initial camera distance, azrollerth, elevation
                # These values are relative to the tracked body and can still be adjusted by the user in the viewer
                viewer.cam.distance = 3.0  # Distance from the target
                viewer.cam.azimuth = 90    # Horizontal angle
                viewer.cam.elevation = -10 # Vertical angle

                print("\nMuJoCo roller Started. Press ESC or close window to exit.")
                print("Actuator Target Velocities (rad/s):")

                # roller loop
                
                boot_time = data.time
                last_print_time = data.time
                prev_time = data.time
                print(boot_time)
                while viewer.is_running():

                    current_time = data.time
                    dt = current_time - prev_time if prev_time is not None else model.opt.timestep
                    if dt <= 0:
                        dt = model.opt.timestep
                    prev_time = current_time

                    if (current_time - last_print_time) >= 0.01:
                        do_print = True
                        last_print_time = current_time
                        print("--------------------------------------------------------------------------------")
                        print(label_line)
                    else:
                        do_print = False

                    

                    for name, [L_id, R_id,sens_id, accel_name,vel_name, left_name, right_name, velocity, left_torque_name, right_torque_name, commands, torques] in bugs.items():

                        accel = data.sensor(accel_name).data
                        vel = data.sensor(vel_name).data
                        left_w = float(data.sensor(left_name).data[0])
                        right_w = float(data.sensor(right_name).data[0])
                        left_tau = float(data.sensor(left_torque_name).data[0])
                        right_tau = float(data.sensor(right_torque_name).data[0])
                        avg_tau = 0.5 * (left_tau + right_tau)
                        torque_log[name].append((left_tau, right_tau))
                        gyro_log[name].append(tuple(vel))
                        command = commands[current_pos]
                        
                        vel = data.sensor(vel_name).data
                        accel = accel/np.linalg.norm(accel)
                        angle = np.dot(accel, grav)

                        base_speed = 300
                        

                        speed = base_speed * command
                        
                        data.ctrl[L_id] = speed
                        data.ctrl[R_id] = speed

                        #look for torque spike

                        # print(speed)
                        if do_print:
                            temp0 = np.concatenate((accel, vel))
                            
                            temp1 = [angle, speed, left_w, right_w, left_tau, right_tau]

                            # temp = np.concatenate((temp0,temp1))
                            # printable = np.round(np.concatenate((temp, [speed])),3)

                            # printable = np.round(np.concatenate((temp0, [speed])),3)
                            printable = np.round(np.concatenate((temp0, temp1)),3)

                            value_line = " ".join(f"{val:>7.3g}" for val in printable)
                            print(value_line)

                    if do_print:
                        do_print = False

                    current_pos = (current_pos + 1)%command_history_length #wrap around code
                    
                    # Step the roller forward
                    mujoco.mj_step(model, data)

                    
                    viewer.sync()

                
                    time.sleep(0.001)
                    pass
        finally:
            plot_sensor_history(torque_log, gyro_log)
            
    except Exception as e:
        print(f"An error occurred during roller: {e}")

if __name__ == "__main__":
    # Call the roller function when the script is executed
    roller_actuator_rotation()
