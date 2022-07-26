import cv2
import numpy as np
import imutils
from myutils import coords_sort


class Contour:

    def __init__(self, img_path, edge_path):
        self.img_path = img_path
        self.edge_path = edge_path
        self.image = None
        self.edges = None
        self.contours = None

    def read_image(self):
        self.image = cv2.imread(self.img_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.edges = cv2.imread(self.edge_path, 0)

    def image_preprocessing(self):
        # Blur
        blur = cv2.GaussianBlur(self.edges,(7,7), 0)
        # Otsu Threshold
        _, self.threshold = cv2.threshold(blur,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.dilation = cv2.dilate(self.threshold, kernel, iterations=1)
        # Canny Edge Detection
        (mu, sigma) = cv2.meanStdDev(self.dilation)
        self.canny = cv2.Canny(self.dilation, (mu[0][0] - sigma[0][0]), (mu[0][0] + sigma[0][0]))
        # Dilation
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
        self.dilation2 = cv2.dilate(self.canny, kernel2, iterations=1)
        
    def get_contour(self):
        cnts = cv2.findContours(self.dilation2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
        # loop over the contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                area = cv2.contourArea(c)
                if area < (self.dilation2.shape[0]*self.dilation2.shape[1]) and area > (self.dilation2.shape[0]*self.dilation2.shape[1])/2:
                    coord_list = [i[0].tolist() for i in list(approx)]
                    first, second, third, last = coords_sort(coord_list)

                    maxWidth = 1540
                    maxHeight = 1000

                    pts1 = np.float32([first, second, third, last])
                    pts2 = np.float32([[0, 0], [maxWidth, 0], [0, maxHeight], [maxWidth, maxHeight]])
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    imgWarpColored = cv2.warpPerspective(self.image, matrix, (maxWidth, maxHeight))
                    cv2.imwrite("contourResults/"+self.img_path.split("/")[1], cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2RGB))
                    break

    def run(self):
        self.read_image()
        self.image_preprocessing()
        self.get_contour()
