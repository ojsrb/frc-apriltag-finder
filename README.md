# FRC Match video apriltag localizer
Uses a match video to estimate the pose of the camera to be used to automatically track robot actions and positions.

## Current state
Currently only identifies apriltags found, will display their positions on video feed and output their positions.
It is also common that it simply does not find any tags, but I assure you it is running properly.

## Usage
1. Clone the repo


2. `uv sync` to install dependencies


3. `uv run main.py test.mp4` to execute the script on a video.  
This will take a minute to start up, as this is a large project.


4. Detected tags are displayed, identified tags are saved in `output`

## Future features
- Average tag positions to get more accurate locations (WIP)
- Estimate camera specifications (focal dist, distortion)
- Estimate pose of camera
