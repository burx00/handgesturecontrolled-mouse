# AI Virtual Mouse — Hand Gesture & Face Tracking

A real-time computer vision project that transforms hand gestures into mouse controls while tracking facial landmarks and expressions using Python, OpenCV, and MediaPipe.

Control your computer without touching a physical mouse. Using only a webcam, the system detects hand movements, recognizes gestures, tracks facial features, and translates movements into desktop interactions.

## Features

- **Virtual Mouse Control** — Move your cursor using hand movements.
- **Gesture Recognition** — Perform clicks, double-clicks, dragging, right-clicks, and scrolling.
- **Facial Landmark Tracking** — Detect and visualize hundreds of facial landmark points in real time.
- **Facial Expression Detection** — Detect smiling, blinking, and mouth-opening gestures.
- **Face Mesh Visualization** — Display facial tracking points, outlines, and feature connections.
- **Interactive Dashboard** — View hand gestures, facial expressions, and tracking status.
- **Real-Time Processing** — Webcam-based interaction without additional hardware.

## Hand Gesture Controls

| Gesture | Action |
|---|---|
| Move your hand | Move the mouse cursor |
| Pinch thumb + index finger briefly | Left click |
| Perform two quick pinches | Double click |
| Hold thumb + index pinch and move | Drag an object |
| Release the pinch | Drop the object |
| Pinch thumb + middle finger | Right click |
| Extend index + middle fingers and move vertically | Scroll |

## Face Tracking

The system uses MediaPipe Face Landmarker to detect facial landmarks and analyze basic facial movements.

**Supported features:**
- Facial mesh and landmark visualization
- Eye tracking and blink detection
- Smile detection
- Mouth-opening detection
- Face outline and feature tracking

Press **F** to switch between full face mesh, landmark points, and no overlay.

*Face tracking detects facial features and expressions; it does not identify individuals.*

## Keyboard Controls

| Key | Function |
|---|---|
| `S` | Activate mouse control |
| `P` | Pause mouse control |
| `F` | Switch facial tracking visualization |
| `Q` | Quit the application |

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI
- macOS Quartz

## Platform Support

Currently developed for **macOS**, using Quartz for native mouse interactions.

Camera and Accessibility permissions may be required.

## About

An experimental human-computer interaction project exploring how computer vision can replace traditional input devices through natural gestures and facial tracking.

Built with Python, OpenCV, and MediaPipe.
