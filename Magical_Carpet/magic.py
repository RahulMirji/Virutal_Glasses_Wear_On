import cv2
import numpy as np
import time

# --- Part 1: Capture the Reference Background ---
# Give your camera a few seconds to adjust to the lighting.
print("Get ready to capture the background. Do not stand in front of the camera.")
time.sleep(3)
print("Capturing background...")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Capture a few frames to make sure the camera has adjusted
for i in range(30):
    ret, background = cap.read()

if not ret:
    print("Error: Could not capture a background image. Check your camera connection.")
    exit()

# Flip the background for a mirror-like effect, matching the real-time feed
background = cv2.flip(background, 1)

print("Background captured successfully. You can now use the cloth.")

# --- Part 2: The Invisibility Loop ---

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a more natural selfie-view
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the color range for the cloth you are using.
    # We will use a red cloth as an example here.
    # Adjust these values based on your specific cloth color.
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    
    # Create a mask to detect the color
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    # For red, you might need a second range because of how HSV wraps around
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask1 + mask2
    
    # Morphological operations to improve the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    
    # Create the inverse mask
    mask_inv = cv2.bitwise_not(mask)
    
    # Get the part of the background that should be visible
    res1 = cv2.bitwise_and(background, background, mask=mask)
    
    # Get the part of the current frame that should be visible (everything but the cloth)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    # Combine the two parts to create the final effect
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    
    # Display the final output
    cv2.imshow('Invisibility Cloak', final_output)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()