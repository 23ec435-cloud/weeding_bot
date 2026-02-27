# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The project name is autonomus weed spraying robot

It uses two pi cam to detect the weeds
with yolov8 model and sprays the liquid
on the weeds using two axis 2 gimbal having 4 servos 

It also navigates the track using the lenovo usb cam 

# Electronics Components used
- 1 pi 5 8gb ram
- 2 pi cam 3
- 1 lenovo usb cam 
- 1 Arduino uno
- 2 2 chenal motor drivers
- 4 servos (2 per gimbal, one gimbal on each side — left and right)
- 4 D.C motors
- 2 water pump
- 2 relays
- other components will be added such as ai hat+

# Flow chart of the robot
all the image processing is done on the pi and the controle signal to controle the motors , servo and pump is sent to arduino via serial comunication

# Development Environment

Virtual environment is at `venv/` inside the project root (created with `--system-site-packages`).

**Activate:**
```bash
source "venv/bin/activate"
```

**Installed libraries:**
- torch 2.10.0 (CPU)
- ultralytics 8.4.18 (YOLOv8)
- opencv 4.13.0
- pyserial 3.5
- numpy 2.4.2
- picamera2 (via system packages)


# must dos
- every time a code is changed or created put it to gethub repo