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

    def __init__(self):
        # Learning result model file path
        self.faceCascadePath = "./models/opencv/haarcascade_frontalface_alt2.xml"
        self.faceLandmarksPath = "./models/dlib/shape_predictor_68_face_landmarks.dat"
        
        # Learning model
        self.faceCascade = cv2.CascadeClassifier(self.faceCascadePath)
        self.faceLandmarksCascade = dlib.shape_predictor(self.faceLandmarksPath)
        
        # Drawing color
        self.faceColor = (255, 255, 255)
        self.msgColor = (0, 0, 255)

        # Minimum buffer size required for detection sleepiness 
        self.bufferSize = 50
        self.requiredBufferSize = 30
        self.SleepinessEARThreshold = 0.58

        # EAR buffer
        # Using for detection sleepiness
        self.EARbuffer = Buffer(self.bufferSize)

    def getDetectResultFrame(self, frame):
        """ Get the camera frame of the detection sleepiness result.

        Args:
            frame (ndarray): Camera frame.

        Returns:
            ndarray: The camera frame of the detection sleepiness result.
        """
        frame, _ = self.__detection(frame, True)
        return frame

    def isSleepy(self, frame):
        """ Get detection sleepiness result.

        Args:
            frame (ndarray): Camera frame.

        Returns:
            bool: detection sleepiness result. True or False.
        """
        _, ret = self.__detection(frame, False)
        return ret

    def __detection(self, frame, isDrawing):
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
        rect = self.faceCascade.detectMultiScale(frame,
                    scaleFactor=1.11, minNeighbors=3, minSize=(200, 200))
        
        if len(rect) > 0:
            # resize to face size
            # convert frame to dlib rectangle
            resizedFace = self.__resizeFace(frame, rect)
            faceDlibRectangle = dlib.rectangle(0, 0, resizedFace.shape[1], resizedFace.shape[0])

            # caltulation EAR
            # detect sleepiness
            left_EAR, right_EAR = self.__getEARs(resizedFace, faceDlibRectangle)
            isSleepy = self.__detectSleepiness(left_EAR, right_EAR)

            # drawing result
            if isDrawing:
                # drawing a square around the face
                x, y, w, h = rect[0,:]
                cv2.rectangle(frame, (x, y), (x+w, y+h), self.faceColor)
                # drawing left & right EAR(eyes aspect ratio)
                cv2.putText(frame,"leftEAR:{}".format(round(left_EAR,2)),
                        (10,30), cv2.FONT_HERSHEY_DUPLEX, 1, self.msgColor, 1, 1)
                cv2.putText(frame,"rightEAR:{}".format(round(right_EAR,2)),
                        (220,30), cv2.FONT_HERSHEY_DUPLEX, 1, self.msgColor, 1, 1)
                # drawing sleepiness result
                if isSleepy:
                    cv2.putText(frame,"Look sleepy!",
                        (10,70), cv2.FONT_HERSHEY_DUPLEX, 1, self.msgColor, 1, 1)
        else:
            # extract the contents of the buffer if it is not detected
            self.EARbuffer.pop()

        return frame, isSleepy

    def __detectSleepiness(self, left_EAR, right_EAR):
        """ Detect sleepiness from EAR.

        Args:
            left_EAR (float64): Left eye EAR.
            right_EAR (float64): Right eye EAR.

        Returns:
            bool: Detection sleepiness result.
        """
        ret = True
        self.EARbuffer.push(left_EAR + right_EAR)
        if self.EARbuffer.size() >= self.requiredBufferSize and \
            self.EARbuffer.getAvg() > self.SleepinessEARThreshold:
            ret = False

        return ret

    def __getEARs(self, frame, face):
        """ Calculate and get the EAR(eyes aspect ratio).

        Args:
            frame (numpy.ndarray): Camera frame.
            face (rectangle): Face rectangle.

        Returns:
            left_EAR (float64): Left eye EAR.
            right_EAR (float64): Right eye EAR.
        """
        rect = self.faceLandmarksCascade(frame, face)
        rect = face_utils.shape_to_np(rect)

        left_EAR = self.__calcEAR(rect[42:48])
        right_EAR = self.__calcEAR(rect[36:42])

        return left_EAR, right_EAR

    def __calcEAR(self, eye):
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

    def __resizeFace(self, frame, range):
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