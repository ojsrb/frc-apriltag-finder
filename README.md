# FRC Match video apriltag localizer
Uses a match video to estimate the pose of the camera to be used for frc1710's computer vision autoscouting system.

## Current state
Currently only identifies apriltags found, will display their positions on video feed and output their positions.

## Usage
1. Clone the repo
2. `uv sync` to install dependencies
3. `uv run main.py [path/to/video.mp4]` to execute the script on a video, will analyze by frame

## Future features
- Average tag positions to get more accurate locations (WIP)
- Estimate camera specifications (focal dist, distortion)
- Estimate pose of camera