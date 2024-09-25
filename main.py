# import cv2
# import numpy as np

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Give some time to adjust to lighting conditions
# print("Adjusting camera, please wait...")
# cv2.waitKey(2000)

# # Read the initial frame to set as background
# ret, background = cap.read()
# if not ret:
#     print("Error: Could not read from webcam.")
#     cap.release()
#     exit()

# # Convert the background frame to grayscale
# background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
# background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

# print("Background captured. You can start throwing darts.")

# while True:
#     # Capture the current frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break

#     # Convert the current frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (21, 21), 0)

#     # Compute the absolute difference between the background and current frame
#     diff = cv2.absdiff(background_gray, gray)

#     # Apply a threshold to highlight differences
#     _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

#     # Optional: Dilate the threshold image to fill in holes, making the darts more visible
#     thresh = cv2.dilate(thresh, None, iterations=2)

#     # Display the thresholded image
#     cv2.imshow("Dart Detection", thresh)
#     cv2.imshow("Cam", frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Give some time to adjust to lighting conditions
# print("Adjusting camera, please wait...")
# cv2.waitKey(4000)

# # Read the initial frame to set as background
# ret, background = cap.read()
# if not ret:
#     print("Error: Could not read from webcam.")
#     cap.release()
#     exit()

# # Convert the background frame to grayscale
# background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
# background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

# print("Background captured. You can start throwing darts.")

# while True:
#     # Capture the current frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break

#     # Convert the current frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (21, 21), 0)

#     # Compute the absolute difference between the background and current frame
#     diff = cv2.absdiff(background_gray, gray)

#     # Apply a threshold to highlight differences
#     _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

#     # Optional: Dilate the threshold image to fill in holes, making the darts more visible
#     thresh = cv2.dilate(thresh, None, iterations=2)

#     # Find contours in the threshold image
#     contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Draw contours around detected darts
#     for contour in contours:
#         if cv2.contourArea(contour) < 500:  # Filter out small contours that might be noise
#             continue

#         # Get the bounding box for each contour
#         (x, y, w, h) = cv2.boundingRect(contour)
        
#         # Draw the bounding box around the detected dart
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
#         # Optionally, draw the contour itself
#         cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

#     # Display the frame with contours
#     cv2.imshow("Background", background)
#     cv2.imshow("Dart Detection", frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# def capture_frame():
#     # Capture a frame from the webcam
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not capture frame.")
#         return None
#     return frame

# def process_and_show_difference(background, current_frame):
#     # Convert frames to grayscale and blur
#     background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
#     background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)
    
#     current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
#     current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)

#     # Compute the absolute difference between the background and current frame
#     diff = cv2.absdiff(background_gray, current_gray)

#     # Apply a threshold to highlight differences
#     _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

#     # Optional: Dilate the threshold image to fill in holes, making the darts more visible
#     thresh = cv2.dilate(thresh, None, iterations=2)

#     # Find contours in the threshold image
#     contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Draw contours around detected darts
#     for contour in contours:
#         if cv2.contourArea(contour) < 500:  # Filter out small contours that might be noise
#             continue

#         # Get the bounding box for each contour
#         (x, y, w, h) = cv2.boundingRect(contour)
        
#         # Draw the bounding box around the detected dart
#         cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
#         # Optionally, draw the contour itself
#         cv2.drawContours(current_frame, [contour], -1, (0, 255, 0), 2)

#     # Display the frame with contours
#     cv2.imshow("Dart Detection", current_frame)

# # Initial background capture
# print("Adjusting camera, please wait...")
# cv2.waitKey(2000)
# background = capture_frame()
# if background is None:
#     cap.release()
#     exit()

# print("Background captured. Throw darts and press 's' to detect.")

# while True:
#     # Display the live video feed
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break

#     cv2.imshow("Live Feed", frame)

#     # Check for user input to perform subtraction or exit
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):  # Press 'q' to quit
#         break
#     elif key == ord('s'):  # Press 's' to perform image subtraction
#         current_frame = capture_frame()
#         if current_frame is not None:
#             process_and_show_difference(background, current_frame)

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Initialize the background subtractor
# background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# print("Adjusting camera, please wait...")
# cv2.waitKey(2000)

# print("Background modeling started. Throw darts and press 's' to detect.")

# while True:
#     # Capture the current frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break

#     # Apply the background subtractor to get the foreground mask
#     fg_mask = background_subtractor.apply(frame)

#     # Apply some morphological operations to reduce noise in the foreground mask
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

#     # Find contours in the foreground mask
#     contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Draw contours around detected darts and look for triangles
#     for contour in contours:
#         # if cv2.contourArea(contour) < 500:  # Filter out small contours that might be noise
#         #     continue

#         # Approximate the contour to a simpler shape
#         epsilon = 0.04 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)

#         # Check if the approximated contour has 3 vertices (indicating a triangle)
#         if len(approx) == 3:
#             # Draw the triangle in blue on the mask
#             cv2.drawContours(fg_mask, [approx], 0, (255), 3)

#             # Optionally, label the triangle on the frame
#             x, y, w, h = cv2.boundingRect(approx)
#             cv2.putText(frame, 'Triangle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#         else:
#             # Draw bounding box around other contours that aren't triangles
#             (x, y, w, h) = cv2.boundingRect(contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
#             # Optionally, draw the contour itself
#             cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

#     # Display the live feed with detections and the foreground mask with detected triangles
#     cv2.imshow("Live Feed", frame)
#     #cv2.imshow("Foreground Mask", fg_mask)
    
#     # Check for user input to show the detection result or exit
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):  # Press 'q' to quit
#         break
#     elif key == ord('s'):  # Press 's' to show the detection result
#         # Show the thresholded mask and the current frame with contours
#         cv2.imshow("Foreground Mask", fg_mask)

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Check if the webcam is opened correctly
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# # Initialize the background subtractor
# background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# print("Adjusting camera, please wait...")
# cv2.waitKey(2000)

# print("Background modeling started. Dart detection is active.")

# while True:
#     # Capture the current frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Failed to capture frame.")
#         break

#     # Apply the background subtractor to get the foreground mask
#     fg_mask = background_subtractor.apply(frame)

#     # Apply some morphological operations to reduce noise in the foreground mask
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
#     fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)


#     # Find contours in the foreground mask
#     contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     dart_detected = False  # Flag to check if a dart is detected

#     # Draw contours around detected darts and look for triangles
#     for contour in contours:
#         if cv2.contourArea(contour) < 1000:  # Filter out small contours that might be noise
#             continue

#         # Approximate the contour to a simpler shape
#         epsilon = 0.04 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)

#         # Check if the approximated contour has 3 vertices (indicating a triangle)
#         if len(approx) == 3:
#             dart_detected = True  # A dart is detected
#             # Draw the triangle in blue on the mask
#             cv2.drawContours(fg_mask, [approx], 0, (255), 3)

#             # Optionally, label the triangle on the frame
#             x, y, w, h = cv2.boundingRect(approx)
#             cv2.putText(frame, 'Triangle (Dart)', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         else:
#             # Draw bounding box around other contours that aren't triangles
#             (x, y, w, h) = cv2.boundingRect(contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
#             # Optionally, draw the contour itself
#             cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

#     # If a dart is detected, save the frame for analysis
#     if dart_detected:
#         cv2.imwrite('dart_detected.png', frame)
#         print("Dart detected! Image saved as dart_detected.png")

#     # Display the live feed with detections and the foreground mask with detected triangles
#     cv2.imshow("Live Feed", frame)
    
#     # Check for user input to exit
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):  # Press 'q' to quit
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize the background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()

print("Adjusting camera, please wait...")
cv2.waitKey(2000)

print("Background modeling started. Dart detection is active.")

while True:
    # Capture the current frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Apply the background subtractor to get the foreground mask
    fg_mask = background_subtractor.apply(frame)

    # Apply some morphological operations to reduce noise in the foreground mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))  # Basic noise reduction
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dart_detected = False  # Flag to check if a dart is detected

    # Look for triangles among the detected contours
    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Filter out small contours that might be noise
            continue

        # Approximate the contour to a simpler shape
        epsilon = 0.08 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the approximated contour has 3 vertices (indicating a triangle)
        if len(approx) == 3:
            dart_detected = True  # A dart is detected
            
            # Draw the triangle based on the approximated contour
            cv2.drawContours(frame, [approx], 0, (255, 0, 0), 3)

            # Calculate the centroid of the triangle to represent the dart point
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            # Draw a circle at the centroid to indicate the point of the dart
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(frame, "Dart Point", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # If a dart is detected, save the frame for analysis
    if dart_detected:
        cv2.imwrite('dart_detected.png', frame)
        print("Dart detected! Image saved as dart_detected.png")

    # Display the live feed with the detected dart and the triangle
    cv2.imshow("Live Feed", frame)
    cv2.imshow("test", fg_mask)

    # Check for user input to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()