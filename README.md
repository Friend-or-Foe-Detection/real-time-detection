# Real-Time Friend-or-Foe Detection Using Infrared Imagery

![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen) ![License](https://img.shields.io/badge/License-MIT-blue) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)

This repository contains the source code for the Final Year Project: **Real-Time Friend-or-Foe Detection Using Infrared Imagery**, developed by students at **P.E.S. College of Engineering, Mandya**.

## 📖 Abstract
Unmanned vehicles are indispensable for modern surveillance and reconnaissance. However, distinguishing between friendly personnel and potential adversaries in low-light or night-time conditions is a critical challenge. Traditional IFF (Identification Friend or Foe) methods rely on active transponders which introduce complexity, weight, and vulnerability to jamming.

This project introduces a **purely vision-based system leveraging infrared (IR) night-vision imagery and passive IR-reflective markers**. By combining low-cost OV5647 IR-cut cameras with custom retro-reflective IR patches worn by friendly personnel, the system achieves reliable friend-or-foe discrimination under complete darkness, running entirely on an onboard **Raspberry Pi 4 B**.

## 🛠️ Hardware & Tech Stack
*   **Edge Compute:** Raspberry Pi 4 Model B (4 GB RAM)
*   **Vision:** OV5647 5 MP IR-Cut Camera + 3W 850nm IR LED Array
*   **Geolocation:** ESP32 DevKit + NEO-6M GPS Module
*   **Deep Learning:** PyTorch, Ultralytics YOLO (v5s, v8s, 11s)
*   **Ground Station UI:** CustomTkinter & TkinterMapView

## 📁 Project Structure
The repository is split into specific modules to prevent merge conflicts among team members:

- `src/ui/`: Ground Station User Interface and Map Visualization (Team 1)
- `src/core/`: Model inference logic and evaluation tools (Team 2)
- `src/utils/`: Utility scripts for bounding boxes and frame extraction (Team 2)
- `src/hardware/`: ESP32 and Raspberry Pi hardware integration (Project Lead)
- `src/training/`: Training scripts used for our custom YOLOv5s, YOLOv8s, and YOLO11s models.
- `docs/`: Analysis graphs, comparison plots, and training metrics.

*(Note: Heavy model weights (`models/`), raw datasets (`data/`), and test videos (`media/`) are kept locally and are intentionally omitted from GitHub to prevent upload limits).*

## 👥 Authors
*   **D Hariharan** [4PS21IS011]
*   **Gowtham C K** [4PS21IS019]
*   **Sagar H V** [4PS21IS046]

*Under the guidance of **Dr. Minavathi**, Professor and HOD, Dept. of Information Science and Engineering, PESCE Mandya.*

