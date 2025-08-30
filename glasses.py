import cv2
import mediapipe as mp
import numpy as np

# Load the glasses image with transparency
glasses = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Start webcam capture
cap = cv2.VideoCapture(0)

def overlay_transparent(background, overlay, x, y):
    """
    Overlays a transparent image on a background image.
    """
    background_h, background_w, _ = background.shape
    overlay_h, overlay_w, _ = overlay.shape

    # Handle transparency
    if overlay.shape[2] == 4:
        # Separate the alpha channel from the overlay
        overlay_rgb = overlay[:, :, :3]
        overlay_alpha = overlay[:, :, 3] / 255.0

        # Adjust dimensions if the overlay goes out of bounds
        y1, y2 = max(0, y), min(background_h, y + overlay_h)
        x1, x2 = max(0, x), min(background_w, x + overlay_w)

        # Ensure the overlay fits within the background
        overlay_y1, overlay_y2 = y1 - y, y2 - y
        overlay_x1, overlay_x2 = x1 - x, x2 - x

        # Ensure the background region to be replaced has a valid shape
        background_region = background[y1:y2, x1:x2]

        if background_region.size > 0:
            # Resize the overlay and alpha mask to match the background region
            resized_overlay = cv2.resize(overlay_rgb[overlay_y1:overlay_y2, overlay_x1:overlay_x2], 
                                         (background_region.shape[1], background_region.shape[0]))
            resized_alpha = cv2.resize(overlay_alpha[overlay_y1:overlay_y2, overlay_x1:overlay_x2],
                                       (background_region.shape[1], background_region.shape[0]))

            # Blend the two images using the alpha channel
            for c in range(0, 3):
                background_region[:, :, c] = (resized_overlay[:, :, c] * resized_alpha + 
                                               background_region[:, :, c] * (1 - resized_alpha))

            return background
    else:
        # If no alpha channel, simply overlay the image
        background[y:y+overlay_h, x:x+overlay_w] = overlay
        return background


while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image with Face Mesh
    results = face_mesh.process(image)

    # Convert the RGB image back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmark coordinates for the eyes and temples
            left_eye = face_landmarks.landmark[133]
            right_eye = face_landmarks.landmark[362]
            
            # Calculate the distance between the eyes in pixels
            img_h, img_w, _ = image.shape
            left_eye_x = int(left_eye.x * img_w)
            left_eye_y = int(left_eye.y * img_h)
            right_eye_x = int(right_eye.x * img_w)
            right_eye_y = int(right_eye.y * img_h)

            distance = np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)
            
            # Scale the glasses image based on the eye distance
            glasses_width = int(distance * 4.5)  # Adjust this scaling factor as needed
            glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])
            resized_glasses = cv2.resize(glasses, (glasses_width, glasses_height))

            # Calculate the position to place the glasses
            glasses_x = int((left_eye_x + right_eye_x) / 2 - glasses_width / 2)
            glasses_y = int((left_eye_y + right_eye_y) / 2 - glasses_height / 2)

            # Overlay the glasses on the image
            image = overlay_transparent(image, resized_glasses, glasses_x, glasses_y)

    # Display the result
    cv2.imshow('Virtual Glasses Try-On', image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()