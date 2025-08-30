# Virtual Glasses Try-On

An AI-powered application that lets you try on glasses virtually using your webcam. This project uses a real-time computer vision pipeline to detect facial landmarks and seamlessly overlay a glasses frame, dynamically adjusting it to your head's position and orientation.

## ✨ Features

* **Real-time Tracking**: Utilizes Google's MediaPipe FaceMesh to track 468 facial points with high accuracy.
* **Dynamic Overlay**: Automatically scales, positions, and rotates the glasses frame to match your face in real time.
* **Z-depth Simulation**: The glasses resize naturally as you move closer to or further from the camera.
* **Interactive Experience**: Provides a realistic "try-on" feel without needing to wear physical glasses.
* **Easy Setup**: Simple to run with a few Python libraries.

## 💡 How It Works

The core of this project is **MediaPipe FaceMesh**, a robust machine learning solution for face geometry. The script performs the following steps on each frame of the webcam feed:

1.  **Face Landmark Detection**: MediaPipe identifies a set of key facial landmarks, including the corners of the eyes and the bridge of the nose.
2.  **Scaling**: The distance between the eyes is calculated to determine the appropriate size for the glasses. This accounts for the perceived distance from the camera (z-depth).
3.  **Positioning**: The glasses image is centered at a calculated point between the eyes, ensuring they are placed correctly on the face.
4.  **Image Overlay**: A transparent PNG image of glasses is overlaid onto the live video feed at the correct size and position.

## ⚙️ Prerequisites

Make sure you have Python 3.x installed. The project requires the following libraries:

* `opencv-python`
* `mediapipe`
* `numpy`

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```
### 2. Create and activate a virtual environment
This is highly recommended to avoid conflicts with your system's Python packages.
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install the required libraries

With your virtual environment active, install the dependencies:
```bash
pip install opencv-python mediapipe numpy
```
### 4. Add the glasses image

Place a transparent PNG image of glasses named glasses.png in the same directory as the script.

### 5. Run the application
```bash

python3 glasses.py
```

### 📚 Future Enhancements

    Rotation and Tilt: Implement a more advanced algorithm to rotate the glasses based on head tilt.

    Multi-frame Support: Allow users to switch between different glasses frames (e.g., using a dropdown menu or keyboard shortcuts).

    Color Customization: Add a color picker to change the frame color in real time.

    Face Tracking for Multiple People: Extend the system to detect and track multiple faces simultaneously.

### 🤝 Contribution

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.
