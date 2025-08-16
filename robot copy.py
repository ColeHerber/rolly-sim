import mujoco
import numpy as np
import time
from mujoco.viewer import launch_passive
# from scipy.spatial.transform import Rotation as R


# Load the MuJoCo model

def srollylate_actuator_rotation():
    """
    Loads a MuJoCo model from XML, sets target velocities for actuators,
    and runs a srollylation with visualization.
    """
    try:
        # Load the MuJoCo model from the XML string
        model = mujoco.MjModel.from_xml_path("roller/scene.xml")
        data = mujoco.MjData(model)

        # Ensure the model and data are valid
        if model is None:
            print("Error: Could not load MuJoCo model from XML string.")
            return
        
        target_body_name = 'refrence'
        target_body_id = model.body(target_body_name).id


        if target_body_id == -1:
            print(f"Error: Body '{target_body_name}' not found in the model.")
            exit()
        # Define target velocities for each actuator
        # These values are in radians/second, matching the `angle="radian"` compiler setting
       
        bugs = {
            "rolly1": [model.actuator("rolly1_L").id, model.actuator("rolly1_R").id, model.site("rolly1_site").id, "rolly1_accel", "rolly1_gyro", "0.5"],
            "rolly2": [model.actuator("rolly2_L").id, model.actuator("rolly2_R").id, model.site("rolly2_site").id, "rolly2_accel", "rolly2_gyro", "0.5"],
            "rolly3": [model.actuator("rolly3_L").id, model.actuator("rolly3_R").id, model.site("rolly3_site").id, "rolly3_accel", "rolly3_gyro", "0.5"],
        }

        grav = np.array([0,-9.81,0])
        axis = np.array([1, 0, 0])  # rotate around X axis
        angle_degrees = 0

        # Create the rotation object
        # r = R.from_rotvec(np.radians(angle_degrees) * axis)
        # grav = r.apply(grav)
        print(grav)

        labels = [
            "ax", "ay", "az", 
            "vx", "vy", "vz", 
            "speed_g", "speed_v", 
            "speed"
                ]
        label_line = " ".join(f"{label:>7}" for label in labels)





        speed_target = 20
        grav_percent = 10
        vel_percent = -10
        target_percent = -2
        do_print = False
        # Initialize the viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:

            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = target_body_id

                    # Set the camera to tracking mode
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = target_body_id

            # Optional: Adjust initial camera distance, azrollyth, elevation
            # These values are relative to the tracked body and can still be adjusted by the user in the viewer
            viewer.cam.distance = 2.0  # Distance from the target
            viewer.cam.azimuth = 90    # Horizontal angle
            viewer.cam.elevation = -10 # Vertical angle

            print("\nMuJoCo Srollylation Started. Press ESC or close window to exit.")
            print("Actuator Target Velocities (rad/s):")

            # Srollylation loop
            
            start_time = data.time
            print(start_time)
            while viewer.is_running():

                step_start = data.time
                for name, [L_id, R_id,sens_id,accel_name,vel_name, velocity] in bugs.items():
                    # print(pos)
                    if (step_start-start_time) >= 1:
                        do_print = True
                        start_time = step_start
                        print("--------------------------------------------------------------------------------")
                        print(label_line)


                    accel = data.sensor(accel_name).data
                    vel = data.sensor(vel_name).data
                    
                    speed_grav = min(0,speed_target*grav_percent*(np.dot(accel, grav) / (np.linalg.norm(accel)* np.linalg.norm(grav))))
                    speed_vel = speed_target*vel_percent*(np.dot(vel,[0,1,0]))
                    speed = min(0,target_percent*speed_target-speed_grav+speed_vel)


                    # speed = (speed_target-(np.dot(accel, grav) / (np.linalg.norm(accel)* np.linalg.norm(grav))))

                    # speed = 1 - (np.linalg.norm(np.cross(accel, grav)) / (np.linalg.norm(accel) * np.linalg.norm(grav)))
                    # speed = 1 #constant speed
                    data.ctrl[L_id] = np.float64(speed)
                    data.ctrl[R_id] = np.float64(-speed)


                    # print(speed)
                    if do_print:
                        temp0 = np.concatenate((accel, vel))
                        temp1 = [speed_grav, speed_vel]
                        temp = np.concatenate((temp0,temp1))
                        printable = np.round(np.concatenate((temp, [speed])),3)

                        value_line = " ".join(f"{val:>7.3g}" for val in printable)
                        print(value_line)


                    # Set control input for each actuator
                if do_print:
                    do_print = False

                    
                

                # Step the srollylation forward
                mujoco.mj_step(model, data)

                # Update the viewer
                
                viewer.sync()

                # Ensure consistent srollylation time step
                # time_until_next_step = model.opt.timestep - (mujoco.mj_get_current_sensordata(model, data).time - step_start)
                # if time_until_next_step > 0:
                #     pass # You can add a sleep here if needed for real-time pacing,
                #          # but viewer.sync() usually handles frame rate.
                time.sleep(0.001)
                pass
                
    except Exception as e:
        print(f"An error occurred during srollylation: {e}")

if __name__ == "__main__":
    # Call the srollylation function when the script is executed
    srollylate_actuator_rotation()
