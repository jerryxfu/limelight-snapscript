import cv2
import numpy as np


def runPipeline(image, llrobot):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ll_config = True  # Use the values tuned for the Limelight (left) or for the local (right) camera?

    # Define HSV ranges for red and blue cages
    lower_cage_red_m1 = np.array([338, 55, 38]) if ll_config else np.array([330, 55, 40])
    upper_cage_red_m1 = np.array([360, 100, 90]) if ll_config else np.array([360, 100, 90])

    lower_cage_red_m2 = np.array([0, 55, 38]) if ll_config else np.array([0, 55, 40])
    upper_cage_red_m2 = np.array([10, 100, 90]) if ll_config else np.array([10, 100, 90])

    lower_cage_blue = np.array([200, 55, 20]) if ll_config else np.array([210, 100, 40])
    upper_cage_blue = np.array([240, 100, 90]) if ll_config else np.array([240, 100, 100])

    # Convert HSV ranges to OpenCV format
    lower_cage_red_cv_m1 = np.array([int(lower_cage_red_m1[0] / 2), int(lower_cage_red_m1[1] * 2.55), int(lower_cage_red_m1[2] * 2.55)])
    upper_cage_red_cv_m1 = np.array([int(upper_cage_red_m1[0] / 2), int(upper_cage_red_m1[1] * 2.55), int(upper_cage_red_m1[2] * 2.55)])
    lower_cage_red_cv_m2 = np.array([int(lower_cage_red_m2[0] / 2), int(lower_cage_red_m2[1] * 2.55), int(lower_cage_red_m2[2] * 2.55)])
    upper_cage_red_cv_m2 = np.array([int(upper_cage_red_m2[0] / 2), int(upper_cage_red_m2[1] * 2.55), int(upper_cage_red_m2[2] * 2.55)])
    lower_cage_blue_cv = np.array([int(lower_cage_blue[0] / 2), int(lower_cage_blue[1] * 2.55), int(lower_cage_blue[2] * 2.55)])
    upper_cage_blue_cv = np.array([int(upper_cage_blue[0] / 2), int(upper_cage_blue[1] * 2.55), int(upper_cage_blue[2] * 2.55)])

    # Create masks for red and blue cages
    cage_red_mask1 = cv2.inRange(img_hsv, lower_cage_red_cv_m1, upper_cage_red_cv_m1)
    cage_red_mask2 = cv2.inRange(img_hsv, lower_cage_red_cv_m2, upper_cage_red_cv_m2)
    cage_blue_mask = cv2.inRange(img_hsv, lower_cage_blue_cv, upper_cage_blue_cv)

    # Combine the masks
    img_threshold = cv2.bitwise_or(cv2.bitwise_or(cage_red_mask1, cage_red_mask2), cage_blue_mask)
    # img_threshold = cv2.bitwise_or(cage_red_mask1, cage_red_mask2)

    # Find contours
    contours, _ = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largestContour = np.array([[]])  # Empty array of values to send back to the robot
    llpython = []

    if len(contours) > 0:
        min_width_cage = 16
        min_height_cage = 64

        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_width_cage and h >= min_height_cage:
                valid_contours.append(contour)

        if valid_contours:
            cv2.drawContours(image, contours, -1, [255, 255, 255], 1)
            cv2.drawContours(image, valid_contours, -1, [0, 255, 0], 2)

            # Record the largest contour
            largestContour = max(valid_contours, key=cv2.contourArea)

            # Get the axis-aligned bounding box for the contour
            x, y, w, h = cv2.boundingRect(largestContour)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Data to send back to the robot
            llpython = [x, y, w, h]

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
