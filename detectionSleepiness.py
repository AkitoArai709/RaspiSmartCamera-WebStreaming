""" detectionSleepiness.py
    Summay: 
        Perform face recognition and detect sleepiness.
"""

import cv2
import dlib
from buffer import Buffer
from imutils import face_utils
from scipy.spatial import distance

class DetectionSleepiness:
    """ Detection Sleepiness.
        Drawing detection result in image frame.
    """
    # Learning result model file path
    faceCascadePath = "./models/opencv/haarcascade_frontalface_alt2.xml"
    faceLandmarksPath = "./models/dlib/shape_predictor_68_face_landmarks.dat"
    
    # Learning model
    faceCascade = cv2.CascadeClassifier(faceCascadePath)
    faceLandmarksCascade = dlib.shape_predictor(faceLandmarksPath)
    
    # Drawing color
    faceColor = (255, 255, 255)
    msgColor = (0, 0, 255)
    
    # EAR buffer
    # Using for detection sleepiness
    EARbuffer = Buffer(50)

    # Minimum buffer size required for detection sleepiness 
    requiredBufferSize = 30
    SleepinessEARThreshold = 0.58

    @staticmethod
    def getDetectResultFrame(frame):
        """ Get the camera frame of the detection sleepiness result.

        Args:
            frame (ndarray): Camera frame.

        Returns:
            ndarray: The camera frame of the detection sleepiness result.
        """
        frame, _ = DetectionSleepiness.__detection(frame, True)
        return frame

    @staticmethod
    def isSleepy(frame):
        """ Get detection sleepiness result.

        Args:
            frame (ndarray): Camera frame.

        Returns:
            bool: detection sleepiness result. True or False.
        """
        _, ret = DetectionSleepiness.__detection(frame, False)
        return ret

    @staticmethod
    def __detection(frame, isDrawing):
        """ Perform detection.

        Args:
            frame (ndarray): Camera frame.
            isDrawing (bool): Drawing the result to frame.

        Returns:
            frame (ndarray): Camera frame.
            isSleepy (bool): Detection sleepiness result.
        """
        isSleepy = None
        # detect person face
        rect = DetectionSleepiness.faceCascade.detectMultiScale(frame,
                    scaleFactor=1.11, minNeighbors=3, minSize=(200, 200))
        
        if len(rect) > 0:
            # resize to face size
            # convert frame to dlib rectangle
            resizedFace = DetectionSleepiness.__resizeFace(frame, rect)
            faceDlibRectangle = dlib.rectangle(0, 0, resizedFace.shape[1], resizedFace.shape[0])

            # caltulation EAR
            # detect sleepiness
            left_EAR, right_EAR = DetectionSleepiness.__getEARs(resizedFace, faceDlibRectangle)
            isSleepy = DetectionSleepiness.__detectSleepiness(left_EAR, right_EAR)

            # drawing result
            if isDrawing:
                # drawing a square around the face
                x, y, w, h = rect[0,:]
                cv2.rectangle(frame, (x, y), (x+w, y+h), DetectionSleepiness.faceColor)
                # drawing left & right EAR(eyes aspect ratio)
                cv2.putText(frame,"leftEAR:{}".format(round(left_EAR,2)),
                        (10,30), cv2.FONT_HERSHEY_DUPLEX, 1, DetectionSleepiness.msgColor, 1, 1)
                cv2.putText(frame,"rightEAR:{}".format(round(right_EAR,2)),
                        (220,30), cv2.FONT_HERSHEY_DUPLEX, 1, DetectionSleepiness.msgColor, 1, 1)
                # drawing sleepiness result
                if isSleepy:
                    cv2.putText(frame,"Sleepy eyes. Wake up!",
                        (10,70), cv2.FONT_HERSHEY_DUPLEX, 1, DetectionSleepiness.msgColor, 1, 1)
        else:
            # Extract the contents of the buffer if it is not detected
            DetectionSleepiness.EARbuffer.pop()

        return frame, isSleepy

    @staticmethod
    def __detectSleepiness(left_EAR, right_EAR):
        """ Detect sleepiness from EAR.

        Args:
            left_EAR (float64): Left eye EAR.
            right_EAR (float64): Right eye EAR.

        Returns:
            bool: Detection sleepiness result.
        """
        ret = True
        DetectionSleepiness.EARbuffer.push(left_EAR + right_EAR)
        if DetectionSleepiness.EARbuffer.size() >= DetectionSleepiness.requiredBufferSize and \
            DetectionSleepiness.EARbuffer.getAvg() > DetectionSleepiness.SleepinessEARThreshold:
            ret = False

        return ret

    @staticmethod
    def __getEARs(frame, face):
        """ Calculate and get the EAR(eyes aspect ratio).

        Args:
            frame (numpy.ndarray): Camera frame.
            face (rectangle): Face rectangle.

        Returns:
            left_EAR (float64): Left eye EAR.
            right_EAR (float64): Right eye EAR.
        """
        rect = DetectionSleepiness.faceLandmarksCascade(frame, face)
        rect = face_utils.shape_to_np(rect)

        left_EAR = DetectionSleepiness.__calcEAR(rect[42:48])
        right_EAR = DetectionSleepiness.__calcEAR(rect[36:42])

        return left_EAR, right_EAR

    @staticmethod
    def __calcEAR(eye):
        """ Calculate the EAR(eyes aspect ratio)
            EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||).

        Args:
            eye (ndarray): Eye landmarks.

        Returns:
            float64: EAR(eyes aspect ratio).
        """
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        eye_ear = (A + B) / (2.0 * C)
        return round(eye_ear, 3)

    @staticmethod
    def __resizeFace(frame, range):
        """ Resize camera frame.

        Args:
            frame (ndarray): Camera frame.
            range (ndarray): Face range.

        Returns:
            ndarray: Resized camera frame.
        """
        # Since the face detection range is small, increase the range. 
        x, y, w, h = range[0,:]
        w -= 10
        y += 10
        h += 10
        w += 10
        face = frame[y :(y + h), x :(x + w)]
        scale = 480 / h
        return cv2.resize(face, dsize=None, fx=scale, fy=scale)