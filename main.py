import mujoco
import mujoco.viewer
import time

def main():
    # Load the master scene from the models directory
    model = mujoco.MjModel.from_xml_path('models/main_scene.xml')
    data = mujoco.MjData(model)

    # Initialize the official passive viewer
    print("Launching simulation... Press ESC in the viewer to close.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Simulation loop
        while viewer.is_running():
            # Step the simulation forward
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Keep real-time sync
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
