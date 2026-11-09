# Livestream + CV Pipeline
import cv2 as cv
import mediapipe as mp

# OS
import sys
import time

# 3D Rendering
import pyvista as pv
from pyvista import examples
import pyvistaqt as pvqt

# Extra
from scipy.spatial.transform import Rotation as R
import numpy as np


# Variable acronyms
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


## 3D Rendering
# Plot on qt background plotter
pl = pvqt.BackgroundPlotter()
pl.enable_anti_aliasing()
pl.background_color="black"

# Add axes
axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
axes.origin = (0, 0, 0)
pl.add_axes()

# Add mesh (change to)
mesh = examples.download_bunny()
actor = pl.add_mesh(mesh, color="lightblue", show_edges=True) # Actor -> instance of mesh in plotter

# Set pivot point to object's center
actor.origin = actor.center

# Set camera to face mesh
pl.camera_position = 'xy'

# Orientation method
def set_orientation(orientation):
    actor.orientation = orientation
    pl.render()

# Convert hand landmarks into quaternion rotation coords
def hand_rotation(hand):
    # Convert MediaPipe axes (y down, z away) to PyVista's (y up, z toward you)
    wrist, index, pinky = (np.array([hand[i].x, -hand[i].y, -hand[i].z]) for i in (0, 5, 17)) 
    
    # build 3 orthogonal unit vectors attached to the palm
    x = index - pinky
    x/= np.linalg.norm(x)
    
    z = np.cross(x, (index + pinky) / 2 - wrist)
    z/= np.linalg.norm(z)
    
    y = np.cross(z, x)
    
    return R.from_matrix(np.column_stack((x, y, z))) # Construct quaternion rotation for hand

def smooth_rotation(prev, new, alpha=0.3): # Same idea as EMA
    delta = new * prev.inv() # rotation from prev to new in the shortest path 
    return R.from_rotvec(alpha * delta.as_rotvec()) * prev # return applied rotation 


smoothed = R.identity() # bunny starts at its original angle
rest = R.from_euler("y", 180, degrees=True) # fixed neutral pose: upright right hand, palm facing camera (use R.identity() for left hand)

### Hand Landmark Detection
latest_result=None

# output of hand landmarker passed to this method (latest result):
def on_result(result, output_img: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Hand landmarker parameters:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=on_result
)

# FPS
frame_count = 0
start_time = time.perf_counter()

# Start livestream capture
cap = cv.VideoCapture(1) # Set to your webcam, usually "0"

# Display 3d Model
pl.show()

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        # fps counter stuff
        frame_count+=1
        elapsed = time.perf_counter() - start_time
        
        ret, frame = cap.read()
        if not ret:
            break

        # flip frame
        frame = cv.flip(frame, 1)
        
        # convert to RGB and mp.Image for landmark detection
        mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            )
        
        # Async detection
        landmarker.detect_async(mp_image, int(elapsed * 1000))
        
        # Snapshot: the callback thread can replace latest_result at any time
        result = latest_result
        
        # Add landmarks (OPTIONAL)
        if result and result.hand_landmarks:
            h, w = frame.shape[:2]
            for hand in result.hand_landmarks:  
                
                # Reference Points
                p1 = (hand[4].x, hand[4].y, hand[4].z) 
                p2 = (hand[20].x, hand[20].y, hand[20].z)
                p3 = (hand[0].x, hand[0].y, hand[0].z)
                
                # Thumb
                cv.circle(frame, (int (p1[0] * w), int (p1[1] * h)), 4, (255, 100, 0), 3)
                cv.putText(frame, f"({p1[0]:.3f}, {p1[1]:.3f}, {p1[2]:.3f})", (int(p1[0]*w), int(p1[1]*h)), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            
                # Pinky
                cv.circle(frame, (int (p2[0] * w), int (p2[1] * h)), 4, (255, 100, 0), 3)
                cv.putText(frame, f"({p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f})", (int(p2[0]*w), int(p2[1]*h)), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                
                # Base
                cv.circle(frame, (int (p3[0] * w), int (p3[1] * h)), 4, (255, 100, 0), 3)
                cv.putText(frame, f"({p3[0]:.3f}, {p3[1]:.3f}, {p3[2]:.3f})", (int(p3[0]*w), int(p3[1]*h)), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            
            # Rotated hand result
            rot = hand_rotation(result.hand_world_landmarks[0])
            
            # Smooth method, makes rotation in video less jumpy
            smoothed = smooth_rotation(smoothed, rot * rest.inv())
            
            # Orientation result
            y, x, z = smoothed.as_euler("YXZ", degrees=True) # convert quarternions into euler angles  
            
            set_orientation((-x, -y, z)) # negative x and y because of camera flip and hand direction
            
        # calculate fps on 1-sec moving avg
        if elapsed >= 1: 
                fps = frame_count / elapsed
                frame = cv.putText(frame, f"FPS: {fps:.1f}", (20, 30), cv.FONT_HERSHEY_PLAIN,2, (255, 255, 255))
                    
        # Display the processed frame
        cv.imshow("Camera", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# Close window
cap.release()
cv.destroyAllWindows()
pl.close()
for _ in range(10): # Cell wait before closing else crash
    cv.waitKey(1)  
