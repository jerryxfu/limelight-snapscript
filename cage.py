import cv2
import numpy as np


def runPipeline(image, llrobot):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Define the HSV range
    preview = False  # Should we use the values tuned for the Limelight (left) or for the local (right) camera?

    lower_cage_red_m1 = np.array([330, 55, 40]) if not preview else np.array([330, 55, 40])
    upper_cage_red_m1 = np.array([360, 100, 90]) if not preview else np.array([360, 100, 90])
    lower_cage_red_m2 = np.array([0, 55, 40]) if not preview else np.array([0, 55, 40])
    upper_cage_red_m2 = np.array([10, 100, 90]) if not preview else np.array([10, 100, 90])
    lower_cage_red_cv_m1 = np.array([
        int(lower_cage_red_m1[0] / 2), int(lower_cage_red_m1[1] * 2.55), int(lower_cage_red_m1[2] * 2.55)
    ])
    upper_cage_red_cv_m1 = np.array([
        int(upper_cage_red_m1[0] / 2), int(upper_cage_red_m1[1] * 2.55), int(upper_cage_red_m1[2] * 2.55)
    ])
    lower_cage_red_cv_m2 = np.array([
        int(lower_cage_red_m2[0] / 2), int(lower_cage_red_m2[1] * 2.55), int(lower_cage_red_m2[2] * 2.55)
    ])
    upper_cage_red_cv_m2 = np.array([
        int(upper_cage_red_m2[0] / 2), int(upper_cage_red_m2[1] * 2.55), int(upper_cage_red_m2[2] * 2.55)
    ])
    cage_red_mask1 = cv2.inRange(img_hsv, lower_cage_red_cv_m1, upper_cage_red_cv_m1)
    cage_red_mask2 = cv2.inRange(img_hsv, lower_cage_red_cv_m2, upper_cage_red_cv_m2)

    # Convert the HSV to a binary image by removing any pixels that do not fall within the following HSV Min/Max values
    img_threshold = cv2.bitwise_or(cage_red_mask1, cage_red_mask2)

    contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largestContour = np.array([[]])  # Empty array of values to send back to the robot

    llpython = []
    if len(contours) > 0:
        min_width_cage = 16
        min_height_cage = 32

        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_width_cage and h >= min_height_cage:
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


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    largestContour, image, llpython = runPipeline(frame, None)
    print(llpython)

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
