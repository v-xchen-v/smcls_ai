# including,
# Image Enhancement: 1. histogram 2.

import cv2 as cv
import numpy as np
import math
class ImageEnhence(object):
    def __init__(self):
        pass

    def histogramEqualize(self, im):
        """
        # Histogram Equalization
        histogram Equalize a bgr cvMat
        :param im:
        :return:
        """
        im_hist = im.copy()
        for idx, s in enumerate(cv.split(im)):
            s = cv.equalizeHist(s)
            im_hist[:,:,idx] = s
        return im_hist
        # cv.imshow("before", im)
        # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        # for idx, s in enumerate(cv.split(im)):
        #     im[:,:,idx] = clahe.apply(s)
        # # cv.imshow("after", im)
        # cv.waitKey(0)
        pass

    def Laplace(self, im):
        # cv.imshow("before", im)
        im_Laplace = cv.filter2D(im, cv.CV_8UC3, np.array([[0,-1,0], [-1 , 4, -1], [0, -1, 0]]))
        # im_Laplace = cv.filter2D(im, cv.CV_8UC3, np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]))
        # cv.imshow("after", im)
        # cv.waitKey(0)
        return im_Laplace

    def Log(self, im):
        # cv.imshow("before", im)
        im_log = im.copy().astype(np.float32)
        h , w , c = im_log.shape
        v = 10
        for i in range(h):
            for j in range(w):
                im_log[i][j][0] = math.log(1 + v*im[i][j][0]/255.0, v+1)*255.0
                im_log[i][j][1] = math.log(1 + v*im[i][j][1]/255.0, v+1)*255.0
                im_log[i][j][2] = math.log(1 + v*im[i][j][2]/255.0, v+1)*255.0
        im_log = im_log.astype(np.uint8)
        return im_log
        # cv.imshow("after", im_log)
        # cv.waitKey(0)

    def Gamma(self, im, g = 5.0):
        # cv.imshow("before", im)
        im_gamma = im.copy().astype(np.float32)
        h, w , c = im_gamma.shape
        gamma = g
        for i in range(h):
            for j in range(w):
                im_gamma[i][j][0] = math.pow(im_gamma[i][j][0]/255.0, gamma)*255.0
                im_gamma[i][j][1] = math.pow(im_gamma[i][j][1]/255.0, gamma)*255.0
                im_gamma[i][j][2] = math.pow(im_gamma[i][j][2]/255.0, gamma)*255.0
        im_gamma = im_gamma.astype(np.uint8)
        # cv.imshow("after", im_gamma)
        # cv.waitKey(0)
        return im_gamma

imageEnhenceAPI = ImageEnhence()
if __name__ == "__main__":
    imp = ImageEnhence()
    # imp.histogramEqualize(cv.imread("sample_equalizeHist.jpg", cv.IMREAD_COLOR))
    # imp.Laplace(cv.imread("sample_Laplace.jpg", cv.IMREAD_COLOR))
    # imp.Log(cv.imread("sample_Log.jpg", cv.IMREAD_COLOR))
    imp.Gamma(cv.imread("sample_gamma.jpg", cv.IMREAD_COLOR))
    # cv.imshow("sample", cv.imread("sample.jpg"))
    # im = imp.histogramEqualize(cv.imread("sample.jpg"))
    # cv.imshow("after", im)
    # cv.waitKey(0)
