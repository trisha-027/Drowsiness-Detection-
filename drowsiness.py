import argparse

import cv2
import face_recognition
import numpy as np
from pygame import mixer
from scipy.spatial import distance as dist

MIN_AER = 0.30  # minimum aspect ratio
EYE_AR_COSEC_FRAMES = 10  # eye aspect ratio consecutive frames

COUNTER = 0
ALARM_ON = False

mixer.init()
sound = mixer.Sound('alarm.wav')


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear


ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path alarm.wav file")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())


def main():
    global COUNTER, ALARM_ON
    video_capture = cv2.VideoCapture(0)  # start video capturing
    video_capture.set(14, 320)  # setting the vdo capturing frame
    video_capture.set(14, 240)  # setting the vdo capturing frame

    while True:
        ret, frame = video_capture.read()
        face_landmarks_list = face_recognition.face_landmarks(frame)

        for face_landmarks in face_landmarks_list:
            leftEye = face_landmarks['left_eye']
            rightEye = face_landmarks['right_eye']

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2

            lpts = np.array(leftEye)
            rpts = np.array(rightEye)

            cv2.polylines(frame, [lpts], True, (255, 255, 0), 1)
            cv2.polylines(frame, [rpts], True, (255, 255, 0), 1)

            if ear < MIN_AER:
                COUNTER += 1

                if COUNTER >= EYE_AR_COSEC_FRAMES:
                    try:
                        sound.play()

                    except:
                        pass

                    cv2.putText(frame, "Alert! you are feeling sleepy", (8, 29),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            else:
                COUNTER = 0
                ALARM_ON = False

        cv2.putText(frame, "alert", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Sleep Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
