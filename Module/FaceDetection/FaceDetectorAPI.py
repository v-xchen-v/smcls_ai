# from Module.FaceDetection import FaceDetector
# import FaceDetector
import sys, os
current_path = os.path.dirname(__file__)
module_path = os.path.dirname(current_path)
sys.path.append(module_path)
from FaceDetection import FaceDetector
from ImageProcess.ImageProcess import ImageEnhence
import cv2
import tensorflow as tf

class FaceDetecte(object):
    def __init__(self):
        self.sess = tf.Session()
        self.pnet, self.rnet, self.onet = FaceDetector.create_mtcnn(self.sess, None)

        # init arguments
        self.minsize = 10
        self.factor = 0.709
        self.threshold = [0.6, 0.7, 0.7]

    def showDetectRes(self, mat, detect_res):
        detected_count = detect_res.shape[0]
        print("detect %s faces" % detected_count)
        print(detect_res)  # [[249.31944124 243.85022496 367.12045288 402.7110337    0.99775261]]
        for face_ord in detect_res:
            # print(face_ord)
            left = int(face_ord[0])
            right = int(face_ord[2])
            top = int(face_ord[1])
            bottom = int(face_ord[3])
            cv2.rectangle(mat, pt1=(left, top), pt2=(right, bottom), color=(0, 0, 255), thickness=3)

    def detectImg(self, mat):

        detect_res, _ = FaceDetector.detect_face(mat, self.minsize, self.pnet, self.rnet, self.onet, self.threshold,
                                                      self.factor)
        self.showDetectRes(mat, detect_res)

faceDetectorAPI = FaceDetecte()

if __name__ == "__main__":
    ie = ImageEnhence()
    im = cv2.imread("Dataset/images/sample.png")
    im = ie.histogramEqualize(im)
    im = cv2.bilateralFilter(im, 25, 25*2, 25/2)
    im = ie.histogramEqualize(im)
    faceDetectorAPI.detectImg(im)
    cv2.imshow("detected result", im)
    cv2.waitKey(0)
    import sys
    print(sys.path)