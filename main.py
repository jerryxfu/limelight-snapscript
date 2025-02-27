import cv2
import numpy as np


def runPipeline(image_bgr, llrobot):
    # Convert the input image to the HSV color space
    img_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the color yellow
    lower_yellow = np.array([30, 50, 50])
    upper_yellow = np.array([60, 100, 100])

    lower_yellow_opencv = np.array([
        int(lower_yellow[0] / 2),  # Hue: 0-360 to 0-179
        int(lower_yellow[1] * 2.55),  # Saturation: 0-100% to 0-255
        int(lower_yellow[2] * 2.55)  # Value: 0-100% to 0-255
    ])

    upper_yellow_opencv = np.array([
        int(upper_yellow[0] / 2),  # Hue: 0-360 to 0-179
        int(upper_yellow[1] * 2.55),  # Saturation: 0-100% to 0-255
        int(upper_yellow[2] * 2.55)  # Value: 0-100% to 0-255
    ])

    # Convert the HSV to a binary image by removing any pixels
    # that do not fall within the following HSV Min/Max values
    img_threshold = cv2.inRange(img_hsv, lower_yellow_opencv, upper_yellow_opencv)

    # Find contours in the new binary image
    contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContour = np.array([[]])

    # Initialize an empty array of values to send back to the robot
    llpython = []

    # If contours have been detected, draw them
    if len(contours) > 0:
        cv2.drawContours(image_bgr, contours, -1, 255, 2)
        # Record the largest contour
        largestContour = max(contours, key=cv2.contourArea)

        # Get the unrotated bounding box that surrounds the contour
        x, y, w, h = cv2.boundingRect(largestContour)

        # Draw the unrotated bounding box
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Record some custom data to send back to the robot
        llpython = [x, y, w, h]

    # Return the largest contour for the LL crosshair, the modified image, and custom robot data
    return largestContour, image_bgr, llpython, img_threshold


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run the pipeline on the current frame
    largestContour, image_bgr, llpython, img_threshold = runPipeline(frame, None)
    print(llpython)

    # Display the original and thresholded images side by side
    combined_image = np.hstack((frame, cv2.cvtColor(img_threshold, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("Original and Thresholded Image", combined_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
