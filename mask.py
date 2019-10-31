import time
import edgeiq
import cv2
from mask_helper import transparent_overlay


def main():
    # Load cascade classifiers for face detection:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Load mask to overlay on faces
    groucho_glasses = cv2.imread('groucho_glasses.png', -1)

    # Create the mask for the Groucho Glasses
    groucho_glasses_mask = groucho_glasses[:, :, 3]

    # Convert glasses image to BGR (eliminate alpha channel):
    groucho_glasses = groucho_glasses[:, :, 0:]

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
                groucho_glasses = cv2.resize(groucho_glasses, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
                transparent_overlay(face_glass_roi_color, groucho_glasses)

        # Display the resulting frame
        cv2.imshow('alwaysAI - Halloween Mask Demo', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Release everything:
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()