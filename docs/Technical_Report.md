# Technical Report: Automotive Painting Cell Simulation
**Author**: Divyansh Arzare  
**Project**: Robotics Assessment  

## 1. Introduction
This report details the design and simulation of an automated automotive painting cell using MuJoCo. The system leverages a UR10e industrial robotic manipulator mounted on a linear rail to paint the exterior of a standard car chassis.

## 2. Cell Layout & Physical Architecture
### 2.1 Workspace Dimensions
The painting cell is designed with an exact **12,000 mm x 8,000 mm footprint**, marked by safety boundary posts. The origin `(0,0,0)` of the global coordinate system is placed precisely at the volumetric center of the car body.

### 2.2 Car Body Geometry
The target is a Body-in-White (BIW) car chassis. Using the exact mesh bounds of the provided STL, the scaling factors were dialed in to exactly achieve:
*   **Length (X)**: 4,800 mm
*   **Width (Y)**: 1,900 mm
*   **Height (Z)**: 1,600 mm

The car is elevated 400 mm from the floor (supported by a center AGV wedge).

### 2.3 Robotic Manipulator & Linear Rail
*   **Robot Model**: Universal Robots UR10e.
*   **End-Effector**: A custom spray gun geometry with a Tool Center Point (TCP) defined exactly at the nozzle tip.
*   **Linear Rail**: The robot base is mounted on a 5.4-meter linear rail positioned exactly **1,200 mm** from the side panel of the car. The rail length allows the robot to travel the full 4.8m length of the car plus an additional 300 mm clearance on both ends.

## 3. Kinematics & Coordinate Frames
All system components derive from the `(0,0,0)` origin safely.
*   **Car Center**: `[0, 0, 0]` (Floor is at Z = -0.2m, Car elevation is Z = 0.2m)
*   **Rail Center**: `[0, -2.1, -0.12]`
*   **Slider Joint Limits**: `[-2.7, 2.7]` ensuring the 5.4m stroke.

## 4. Path Planning & Trajectory (Planned)
The trajectory generator (stubbed in `controllers/trajectory_generator.py`) will compute raster (zigzag) paths across the exterior mesh surfaces, utilizing the `STANDOFF_DISTANCE` (250mm) to maintain optimal paint application.

## 5. Collision Avoidance (Planned)
The `collision_checker.py` module will leverage MuJoCo's native collision queries (`data.ncon`) to interrupt execution or correct joint angles if the spray gun intersects with the BIW chassis or the robot self-collides.

## 6. Conclusion
The cell architecture is mathematically clean, modular, and serves as a highly extensible digital twin for automated painting path generation.
