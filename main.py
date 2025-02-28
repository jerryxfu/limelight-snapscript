import cv2
import numpy as np


def runPipeline(_image, llrobot):
    img_hsv = cv2.cvtColor(_image, cv2.COLOR_BGR2HSV)

    # Define the HSV range
    lower_yellow = np.array([155, 35, 20])
    upper_yellow = np.array([180, 75, 100])

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

    # Convert the HSV to a binary image by removing any pixels that do not fall within the following HSV Min/Max values
    img_threshold = cv2.inRange(img_hsv, lower_yellow_opencv, upper_yellow_opencv)

    # Find contours in the new binary image
    contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContour = np.array([[]])

    # Empty array of values to send back to the robot
    llpython = []

    if len(contours) > 0:
        min_width = 50
        min_height = 50
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_width and h >= min_height:
                valid_contours.append(contour)

        if valid_contours:
            cv2.drawContours(_image, contours, -1, [255, 255, 255], 2)
            # Record the largest contour
            largestContour = max(contours, key=cv2.contourArea)

            # Get the axis aligned bounding box
            x, y, w, h = cv2.boundingRect(largestContour)

            # Draw the bounding box
            cv2.rectangle(_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.rectangle(img_threshold, (x, y), (x + w, y + h), (255, 255, 255), 1)

            # Data to send back to the robot
            llpython = [x, y, w, h]

    # Return the largest contour for the LL crosshair, the modified image, and custom robot data
    return largestContour, _image, llpython, img_threshold


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    largestContour, image, llpython, img_threshold = runPipeline(frame, None)
    print(llpython)

    # Display the original and thresholded images side by side
    combined_image = np.hstack((frame, cv2.cvtColor(img_threshold, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("Original and Thresholded Image", combined_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
