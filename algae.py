import cv2
import numpy as np


def runPipeline(image, llrobot):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Define the HSV range
    preview = False  # Should we use the values tuned for the Limelight (left) or for the local (right) camera?

    lower_algae = np.array([155, 35, 20]) if not preview else np.array([150, 35, 20])
    upper_algae = np.array([180, 75, 100]) if not preview else np.array([180, 75, 100])
    lower_algae_cv = np.array([
        int(lower_algae[0] / 2), int(lower_algae[1] * 2.55), int(lower_algae[2] * 2.55)
    ])
    upper_algae_cv = np.array([
        int(upper_algae[0] / 2), int(upper_algae[1] * 2.55), int(upper_algae[2] * 2.55)
    ])

    # Convert the HSV to a binary image by removing any pixels that do not fall within the following HSV Min/Max values
    img_threshold = cv2.inRange(img_hsv, lower_algae_cv, upper_algae_cv)
    contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largestContour = np.array([[]])  # Empty array of values to send back to the robot
    llpython = []
    if len(contours) > 0:
        min_width_algae = 50
        min_height_algae = 50

        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_width_algae and h >= min_height_algae:
                valid_contours.append(contour)
                if valid_contours:
                    # cv2.drawContours(image, contours, -1, [255, 255, 255], 1)

                    # Record the largest contour
                    largestContour = max(contours, key=cv2.contourArea)

                    # Get the axis aligned bounding box
                    x, y, w, h = cv2.boundingRect(largestContour)

                    # Draw the bounding box
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

                    # Data to send back to the robot
                    llpython = [x, y, w, h]  # Return the largest contour for the LL crosshair, the modified image, and custom robot data
    return largestContour, image, llpython
