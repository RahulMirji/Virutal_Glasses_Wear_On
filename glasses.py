import cv2
import mediapipe as mp
import numpy as np
import os

# A list of your glasses image files. Make sure they are in the same folder.
# You will need to download and name them glasses1.png, glasses2.png, etc.
glasses_files = [f'glasses{i}.png' for i in range(1, 11)]
current_glasses_index = 0

# --- Helper Function for Image Overlay ---
def overlay_transparent(background, overlay, x, y):
    background_h, background_w, _ = background.shape
    overlay_h, overlay_w, _ = overlay.shape

    if overlay.shape[2] == 4:
        overlay_rgb = overlay[:, :, :3]
        overlay_alpha = overlay[:, :, 3] / 255.0

        y1, y2 = max(0, y), min(background_h, y + overlay_h)
        x1, x2 = max(0, x), min(background_w, x + overlay_w)

        overlay_y1, overlay_y2 = y1 - y, y2 - y
        overlay_x1, overlay_x2 = x1 - x, x2 - x

        background_region = background[y1:y2, x1:x2]

        if background_region.size > 0:
            resized_overlay = cv2.resize(overlay_rgb[overlay_y1:overlay_y2, overlay_x1:overlay_x2], 
                                         (background_region.shape[1], background_region.shape[0]))
            resized_alpha = cv2.resize(overlay_alpha[overlay_y1:overlay_y2, overlay_x1:overlay_x2],
                                       (background_region.shape[1], background_region.shape[0]))

            for c in range(0, 3):
                background_region[:, :, c] = (resized_overlay[:, :, c] * resized_alpha + 
                                               background_region[:, :, c] * (1 - resized_alpha))
            return background
    else:
        background[y:y+overlay_h, x:x+overlay_w] = overlay
        return background

# --- Main Script ---

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Start webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Load the current glasses image.
    if os.path.exists(glasses_files[current_glasses_index]):
        glasses = cv2.imread(glasses_files[current_glasses_index], cv2.IMREAD_UNCHANGED)
        
        # Add a check to ensure the loaded image is valid and has 3 dimensions.
        if glasses is None or len(glasses.shape) < 3:
            print(f"Warning: Skipping frame due to a bad image file: {glasses_files[current_glasses_index]}")
            continue
    else:
        print(f"Warning: Skipping frame due to missing file: {glasses_files[current_glasses_index]}")
        continue

    # Convert the BGR image to RGB for MediaPipe
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Process the image with Face Mesh
    results = face_mesh.process(image)

    # Convert the RGB image back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmark coordinates for the eyes
            left_eye = face_landmarks.landmark[133]
            right_eye = face_landmarks.landmark[362]
            
            # Calculate the distance between the eyes in pixels
            img_h, img_w, _ = image.shape
            left_eye_x = int(left_eye.x * img_w)
            right_eye_x = int(right_eye.x * img_w)

            distance = right_eye_x - left_eye_x
            
            # Scale the glasses image based on the eye distance
            glasses_width = int(distance * 5)  # Adjust this scaling factor as needed
            glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])
            if glasses_width > 0 and glasses_height > 0:
                resized_glasses = cv2.resize(glasses, (glasses_width, glasses_height))
            else:
                continue

            # Calculate the position to place the glasses
            glasses_x = int(left_eye_x + distance / 2 - glasses_width / 2)
            glasses_y = int(face_landmarks.landmark[168].y * img_h - glasses_height / 2)

            # Overlay the glasses on the image
            image = overlay_transparent(image, resized_glasses, glasses_x, glasses_y)

    # Display the result
    cv2.imshow('Virtual Glasses Try-On', image)

    # Key press handling for switching glasses
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'): # 'n' for next
        current_glasses_index = (current_glasses_index + 1) % len(glasses_files)
    elif key == ord('p'): # 'p' for previous
        current_glasses_index = (current_glasses_index - 1 + len(glasses_files)) % len(glasses_files)

cap.release()
cv2.destroyAllWindows()