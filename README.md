# hand-3d-controller

Rotate a 3D model with your hand. Pipeline feeds OpenCV webcam data into the mediapipe ![Hand Landmarker model](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker). The hand landmarkers are then processed into Quaternions to calculate model orientation, and finally passed to the PyVista visualizer.

## Setup

Requires Python, a webcam, and a Qt binding for the PyVista window (PyQt6 below):

```bash
pip install opencv-python mediapipe pyvista pyvistaqt PyQt6 scipy numpy
```

The hand model, `hand_landmarker.task`, is included in the repo. The demo mesh (the Stanford bunny) is downloaded by PyVista on first run, so that run needs an internet connection.

## Usage

Run from the repo root, since the model is loaded by relative path:

```bash
python main.py
```

Face your hand palm forward to your webcam at about half a meter away. Rotate your hand about your wrist and watch the model follow. 

Press `q` to quit the application.

## Config

Webcam -> `cv.VideoCapture(1)`, use the index of your camera usually 0 or 1

Left hand use -> change `rest = R.fromEuler(...)` to `R.identity()`

3D model -> `mesh = examples.download_bunny()`, load your own, e.g. `pv.read("model.stl")`

Smoothing -> Modify `alpha=0.3`, lower is steadier but lags more, higher is more responsive but jittier


## Credits
Portions of this page are reproduced from work created and shared by Google and used according to terms described in the Creative Commons 4.0 Attribution License. 
