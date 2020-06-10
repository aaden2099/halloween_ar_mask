import copy
import cv2
from mask_helper import transparent_overlay
import time


def main():
    # Load cascade classifiers for face detection:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Load mask to overlay on faces
    mask = cv2.imread('groucho_glasses.png', -1)

    # Convert glasses image to BGR (eliminate alpha channel):
    mask = mask[:, :, 0:]

    # Make a deep copy of the mask
    mask_copy = copy.deepcopy(mask)

    # Create VideoCapture object to get images from the camera:
    video_capture = cv2.VideoCapture(0)

    # Allow Webcam to warm up
    time.sleep(2.0)

    while True:
        # Capture frame from the VideoCapture object:
        ret, frame = video_capture.read()

        # Convert frame to grayscale:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using the function 'detectMultiScale()'
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Iterate over each detected face:
        for (x, y, w, h) in faces:
            if h > 0 and w > 0:
                glass_ymin = int(y)
                glass_ymax = int(y + h)
                sh_glass = glass_ymax - glass_ymin
                face_glass_roi_color = frame[glass_ymin:glass_ymax, int(x):int(x + w)]
                mask_copy = cv2.resize(mask_copy, (w, sh_glass), interpolation=cv2.INTER_LINEAR)
                transparent_overlay(face_glass_roi_color, mask_copy)

        # Display the resulting frame with title
        cv2.imshow('Mask Demo', frame)

        # Deep copy is made so that same mask image isn't reused. Otherwise mask will become pixelated.
        mask_copy = copy.deepcopy(mask)

        # Press q to stop the program (lowercase Q)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when program ends:
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
