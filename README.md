# Automated Automotive Painting Cell

![Simulation Screenshot](docs/screenshots/placeholder.png)

This repository contains the simulation of an industrial robotic painting cell built in **MuJoCo**. It features a Universal Robots UR10e manipulator mounted on a linear rail, operating on a 4.8m x 1.9m car chassis.

## Features
- **Accurate Dimensions**: Car scaled precisely to `4800x1900x1600 mm`.
- **Industrial Layout**: `12m x 8m` floor plan with strict geometric alignment.
- **Linear Rail Integration**: 5.4m rail track enabling full exterior reach with 1200mm side clearance.
- **Custom End-Effector**: Spray gun geometry with distinct Tool Center Point (TCP).

## Project Structure
```plaintext
assets/             # Raw STL files (Car, Robot, Cell)
controllers/        # Control logic and IK wrappers
docs/               # Technical Report and media
models/             # MuJoCo XML files (main_scene, ur10e_rail, cell_layout)
utils/              # Helpers for mesh preparation and visualization
main.py             # Single entry point
config.py           # Global physics/dimension parameters
```

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the simulation:
   ```bash
   python main.py
   ```

## Deliverables
- **Technical Report**: Available in `docs/Technical_Report.md` (and PDF export).
- **Demo Video**: (Record a 3-7 minute screen capture of the MuJoCo viewer showcasing the linear rail movement and attach/upload to Google Drive).

## License
MIT
