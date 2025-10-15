import mujoco
import numpy as np
import time
from mujoco.viewer import launch_passive


# Load the MuJoCo model
def rotate_vector(v, axis, angle_deg):
    """
    Rotate vector v around 'axis' by angle_deg (degrees).
    Uses Rodrigues' rotation formula. Works with 1D or Nx3 array v.
    """
    v = np.asarray(v, dtype=float)
    axis = np.asarray(axis, dtype=float)

    # normalize axis
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Axis must be non-zero.")
    k = axis / norm

    theta = np.deg2rad(angle_deg)
    ct, st = np.cos(theta), np.sin(theta)

    # skew-symmetric cross-product matrix of k
    K = np.array([
        [    0, -k[2],  k[1]],
        [ k[2],     0, -k[0]],
        [-k[1],  k[0],     0]
    ])

    # 3x3 rotation matrix
    R = ct * np.eye(3) + st * K + (1 - ct) * np.outer(k, k)

    # support single vector (3,) or batch (N,3)
    return (R @ v) if v.ndim == 1 else (v @ R.T)


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
        # Define target velocities for each actuator
        # These values are in radians/second, matching the `angle="radian"` compiler setting
       
        bugs = {
            "roller1": [model.actuator("L1").id, model.actuator("R1").id, model.site("roller1").id, "roller1_accel", "roller1_gyro", "L1_w", "R1_w", "0.5"],
            "roller2": [model.actuator("L2").id, model.actuator("R2").id, model.site("roller2").id, "roller2_accel", "roller2_gyro", "L2_w", "R2_w", "0.5"],
            "roller3": [model.actuator("L3").id, model.actuator("R3").id, model.site("roller3").id, "roller3_accel", "roller3_gyro", "L3_w", "R3_w", "0.5"],
            "roller4": [model.actuator("L4").id, model.actuator("R4").id, model.site("roller4").id, "roller4_accel", "roller4_gyro", "L4_w", "R4_w", "0.5"],
            "roller5": [model.actuator("L5").id, model.actuator("R5").id, model.site("roller5").id, "roller5_accel", "roller5_gyro", "L5_w", "R5_w", "0.5"],
            "roller6": [model.actuator("L6").id, model.actuator("R6").id, model.site("roller6").id, "roller6_accel", "roller6_gyro", "L6_w", "R6_w", "0.5"],

        }

        # grav = np.array([0,0,-9.81])
        grav = np.array([0,0,1])

        axis = np.array([1, 0, 0])  # rotate around X axis
        n = np.array([1,0,0])   # x-axis defines the "sign" direction

        # Create the rotation object
        grav = rotate_vector(grav, axis, 45)   # rotate by 60 degrees
        print(grav)

        labels = [
            "ax", "ay", "az", 
            "vx", "vy", "vz", 
             "grav%", "command", "L_w", "R_w", 
                ]
        label_line = " ".join(f"{label:>7}" for label in labels)




        do_print = False
        startup_ramp_duration = 0  # seconds to blend control in; lower is snappier, 0 disables ramp
        filter_time_constant = 0  # seconds; lower = quicker response, higher = smoother output
        # Initialize the viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:

            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = target_body_id

                    # Set the camera to tracking mode
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = target_body_id

            # Optional: Adjust initial camera distance, azrollerth, elevation
            # These values are relative to the tracked body and can still be adjusted by the user in the viewer
            viewer.cam.distance = 2.0  # Distance from the target
            viewer.cam.azimuth = 90    # Horizontal angle
            viewer.cam.elevation = -10 # Vertical angle

            print("\nMuJoCo roller Started. Press ESC or close window to exit.")
            print("Actuator Target Velocities (rad/s):")

            # roller loop
            
            boot_time = data.time
            last_print_time = data.time
            filtered_ctrl = {}
            for L_id, R_id, *_ in bugs.values():
                filtered_ctrl[L_id] = 0.0
                filtered_ctrl[R_id] = 0.0
            prev_time = data.time
            print(boot_time)
            while viewer.is_running():
                current_time = data.time
                dt = current_time - prev_time if prev_time is not None else model.opt.timestep
                if dt <= 0:
                    dt = model.opt.timestep
                prev_time = current_time

                if (current_time - last_print_time) >= 0.1:
                    do_print = True
                    last_print_time = current_time
                    print("--------------------------------------------------------------------------------")
                    print(label_line)
                else:
                    do_print = False

                

                for name, [L_id, R_id,sens_id, accel_name,vel_name, left_name, right_name, velocity] in bugs.items():

                    accel = data.sensor(accel_name).data
                    vel = data.sensor(vel_name).data
                    left_w = float(data.sensor(left_name).data[0])
                    right_w = float(data.sensor(right_name).data[0])

                    
                    # speed_grav = speed_target*grav_percent*(np.dot(accel, grav) / (np.linalg.norm(accel)* np.linalg.norm(grav)))
                    # speed = (speed_target-(np.dot(accel, grav) / (np.linalg.norm(accel)* np.linalg.norm(grav))))
                    
                    vel = data.sensor(vel_name).data
                    # speed = np.linalg.norm(np.cross(accel, grav))
                    accel = accel/np.linalg.norm(accel)
                    # cross = np.cross(accel, grav)
                    angle = np.dot(accel, grav)

                    if angle < 0:
                        ang_speed = 1
                    else:
                        ang_speed = 0.95                   # speed = (angle)/(np.pi)
                    speed_multiplier = 300

                    speed = ((ang_speed)*speed_multiplier)
                    target_ctrl = np.float64(speed)

                    if filter_time_constant > 0:
                        alpha = 1 - np.exp(-dt / filter_time_constant)
                        alpha = np.clip(alpha, 0.0, 1.0)
                        filtered_ctrl[L_id] += alpha * (target_ctrl - filtered_ctrl[L_id])
                        filtered_ctrl[R_id] += alpha * (target_ctrl - filtered_ctrl[R_id])
                        data.ctrl[L_id] = filtered_ctrl[L_id]
                        data.ctrl[R_id] = filtered_ctrl[R_id]
                    else:
                        data.ctrl[L_id] = target_ctrl
                        data.ctrl[R_id] = target_ctrl


                    # print(speed)
                    if do_print:
                        temp0 = np.concatenate((accel, vel))
                        
                        temp1 = [angle, speed, left_w, right_w]

                        # temp = np.concatenate((temp0,temp1))
                        # printable = np.round(np.concatenate((temp, [speed])),3)

                        # printable = np.round(np.concatenate((temp0, [speed])),3)
                        printable = np.round(np.concatenate((temp0, temp1)),3)

                        value_line = " ".join(f"{val:>7.3g}" for val in printable)
                        print(value_line)


                    # Set control input for each actuator
                if do_print:
                    do_print = False

                    
                

                # Step the roller forward
                mujoco.mj_step(model, data)

                # Update the viewer
                
                viewer.sync()

                # Ensure consistent roller time step
                # time_until_next_step = model.opt.timestep - (mujoco.mj_get_current_sensordata(model, data).time - step_start)
                # if time_until_next_step > 0:
                #     pass # You can add a sleep here if needed for real-time pacing,
                #          # but viewer.sync() usually handles frame rate.
                time.sleep(0.0001)
                pass
                
    except Exception as e:
        print(f"An error occurred during roller: {e}")

if __name__ == "__main__":
    # Call the roller function when the script is executed
    roller_actuator_rotation()
