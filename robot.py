import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os
import pandas as pd

def simulate_and_analyze():
    """
    Loads a MuJoCo model, runs a simulation, plots and analyzes the results,
    saves the plot, and appends the results to an Excel file.
    """
    try:
        # Load the MuJoCo model from the XML file
        # Ensure the path to your 'scene.xml' is correct
        model = mujoco.MjModel.from_xml_path("roller/scene.xml")
        data = mujoco.MjData(model)

        if model is None:
            print("Error: Could not load MuJoCo model from XML.")
            return

        # --- Simulation Setup ---
        # Add velocity sensor names back to the dictionary
        bugs = {
            "rolly1": [model.actuator("rolly1_L").id, model.actuator("rolly1_R").id, "rolly1_accel", "rolly1_gyro"],
            "rolly2": [model.actuator("rolly2_L").id, model.actuator("rolly2_R").id, "rolly2_accel", "rolly2_gyro"],
            "rolly3": [model.actuator("rolly3_L").id, model.actuator("rolly3_R").id, "rolly3_accel", "rolly3_gyro"],
        }

        # Set gravity vector (rotated __ degrees around x-axis)
        grav = np.array([0, -9.81, 0])
        axis = np.array([1, 0, 0])
        angle_degrees = 0
        r = R.from_rotvec(np.radians(angle_degrees) * axis)
        grav = r.apply(grav)

        # Control parameters
        speed_target = 0.5
        grav_percent = 6
        vel_percent = 20  # Added velocity percentage
        target_percent = 1.5
        
        # --- Data Collection ---
        simulation_time = []
        x_dot_history = []
        theta_dot_history = []
        
        # Initialize dictionaries to hold speed history for each roller
        L_speed_history = {name: [] for name in bugs.keys()}
        R_speed_history = {name: [] for name in bugs.keys()}
        
        simulation_duration = 15  # seconds
        print(f"Running simulation for {simulation_duration} seconds...")

        # --- Simulation Loop ---
        while data.time < simulation_duration:
            full_vel = data.sensor("total_vel").data
            full_ang = data.sensor("total_ang").data
            
            x_dot = np.linalg.norm(full_vel)
            theta_dot = full_ang[1]

            simulation_time.append(data.time)
            x_dot_history.append(x_dot)
            theta_dot_history.append(theta_dot)

            # --- Control Logic (updated) ---
            for name, [L_id, R_id, accel_name, vel_name] in bugs.items():
                accel = data.sensor(accel_name).data
                vel = data.sensor(vel_name).data

                speed_grav = 0
                if np.linalg.norm(accel) > 1e-6 and np.linalg.norm(grav) > 1e-6:
                    speed_grav = speed_target * grav_percent * (np.dot(accel, grav) / (np.linalg.norm(accel) * np.linalg.norm(grav)))
                
                
                
                speed_vel = speed_target * vel_percent * (np.dot(vel, [0, -1, 0]))
                
                speed = target_percent - speed_grav + speed_vel
                
                # Set control and log the speed setpoints
                data.ctrl[L_id] = np.float64(speed)
                data.ctrl[R_id] = np.float64(-speed)
                L_speed_history[name].append(speed)
                R_speed_history[name].append(-speed)
                
            mujoco.mj_step(model, data)

        print("Simulation finished.")

        # --- Analysis and Plotting ---
        simulation_time = np.array(simulation_time)
        x_dot_history = np.array(x_dot_history)
        theta_dot_history = np.array(theta_dot_history)

        steady_state_start_time = 8.0
        steady_state_indices = np.where(simulation_time >= steady_state_start_time)
        
        avg_x_dot = 0
        avg_theta_dot = 0

        if len(steady_state_indices[0]) > 0:
            steady_state_x_dot = x_dot_history[steady_state_indices]
            steady_state_theta_dot = theta_dot_history[steady_state_indices]
            avg_x_dot = np.mean(steady_state_x_dot)
            avg_theta_dot = np.mean(steady_state_theta_dot)

            print("\n--- Steady-State Analysis (last 2 seconds) ---")
            print(f"Average Forward Velocity (x_dot): {avg_x_dot:.4f} m/s")
            print(f"Average Rotational Velocity (theta_dot): {avg_theta_dot:.4f} rad/s")
        else:
            print("\nCould not perform steady-state analysis: Simulation did not reach 8 seconds.")

        # --- Plotting ---
        # Create 4 subplots vertically
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
        fig.suptitle('Simulation Profiles', fontsize=16)

        # Plot x_dot
        ax1.plot(simulation_time, x_dot_history, label='x_dot (Forward Velocity)', color='b')
        ax1.axhline(y=avg_x_dot, color='g', linestyle='--', label=f'Avg x_dot: {avg_x_dot:.2f} m/s')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title('Forward Velocity (x_dot) vs. Time')
        ax1.grid(True)
        ax1.legend()

        # Plot theta_dot
        ax2.plot(simulation_time, theta_dot_history, label='theta_dot (Rotational Velocity)', color='r')
        ax2.axhline(y=avg_theta_dot, color='g', linestyle='--', label=f'Avg theta_dot: {avg_theta_dot:.2f} rad/s')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.set_title('Rotational Velocity (theta_dot) vs. Time')
        ax2.grid(True)
        ax2.legend()
        
        # Define marker styles for each roller to differentiate them
        markers = ['o', 's', '^']

        # Plot Left Motor Speeds
        for i, (name, speeds) in enumerate(L_speed_history.items()):
            ax3.plot(simulation_time, speeds, label=f'{name} Left Speed', marker=markers[i], markevery=1000, linestyle='-',alpha=0.5)
        ax3.set_ylabel('Speed Setpoint')
        ax3.set_title('Left Motor Speed Setpoints vs. Time')
        ax3.grid(True)
        ax3.legend()

        # Plot Right Motor Speeds
        for i, (name, speeds) in enumerate(R_speed_history.items()):
            ax4.plot(simulation_time, speeds, label=f'{name} Right Speed', marker=markers[i], markevery=1000, linestyle='-',alpha=0.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Speed Setpoint')
        ax4.set_title('Right Motor Speed Setpoints vs. Time')
        ax4.grid(True)
        ax4.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs('graphics', exist_ok=True)
        filename = f"graphics/{grav_percent}_{vel_percent}_{target_percent}_{speed_target}__{angle_degrees}.png"
        plt.savefig(filename)
        print(f"\nPlot saved to {filename}")
        plt.show(block=False)
        plt.pause(1)
        plt.close()

        # --- Excel Logging ---
        excel_filename = 'simulation_results.xlsx'
        new_data = {
            'Grav %': [grav_percent],
            'Vel %': [vel_percent],
            'Target %': [target_percent],
            'Speed Target': [speed_target],
            'Grav Rotate': [angle_degrees],
            'X_dot': [avg_x_dot],
            'Theta_dot': [avg_theta_dot]
        }
        new_df = pd.DataFrame(new_data)

        try:
            existing_df = pd.read_excel(excel_filename)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except FileNotFoundError:
            combined_df = new_df

        combined_df.to_excel(excel_filename, index=False)
        print(f"Results appended to {excel_filename}")

    except Exception as e:
        print(f"An error occurred during simulation: {e}")

if __name__ == "__main__":
    # Ensure you have pandas and openpyxl installed:
    # pip install pandas openpyxl
    simulate_and_analyze()