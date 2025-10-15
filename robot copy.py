import mujoco
import numpy as np
import time
from mujoco.viewer import launch_passive
from collections import deque


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
                    command = commands[current_pos]
                    
                    vel = data.sensor(vel_name).data
                    accel = accel/np.linalg.norm(accel)
                    angle = np.dot(accel, grav)

                    base_speed = 10
                    

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

            
                time.sleep(0.05)
                pass
            
    except Exception as e:
        print(f"An error occurred during roller: {e}")

if __name__ == "__main__":
    # Call the roller function when the script is executed
    roller_actuator_rotation()
