import numpy as np
import cv2
from myutils import segment_by_angle_kmeans, segmented_intersections


class HoughTransform():
    def __init__(self, img_path, edge_path):
        self.img_path = img_path
        self.edge_path = edge_path

    def read_image(self):
        self.image = cv2.imread(self.img_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.edges = cv2.imread(self.edge_path, 0)

    def image_preprocessing(self):
        # Blur
        blur = cv2.GaussianBlur(self.edges,(7,7), 0)
        # Threshold
        # _, thresholded = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        dilation = cv2.dilate(blur, kernel, iterations=1)
        # Canny Edge Detection
        (mu, sigma) = cv2.meanStdDev(dilation)
        self.canny = cv2.Canny(dilation, (mu[0][0] - sigma[0][0]), (mu[0][0] + sigma[0][0]))
        # Dilation
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
        self.dilation2 = cv2.dilate(self.canny, kernel2, iterations=1)

    def hough_lines(self):
        # Hough Transform Lines along the edges
        self.lines = cv2.HoughLines(self.dilation2, rho=0.75, theta=np.pi / 90, threshold=int((self.image.shape[0])/2.1))
        points = []
        if self.lines is not None:
            for i in range(0, len(self.lines)):

                rho = self.lines[i][0][0]
                theta = self.lines[i][0][1]
                a = np.math.cos(theta)
                b = np.math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 4000*(-b)), int(y0 + 4000*(a)))
                pt2 = (int(x0 - 4000*(-b)), int(y0 - 4000*(a)))
                points.append([pt1,pt2])

    def crop_process(self):
        # Segmentation the lines by angle
        segmented = segment_by_angle_kmeans(self.lines)
        # Finding the intersections
        intersections = segmented_intersections(segmented)
        # Finding vertical and horizontal intersection points
        vertical = [i[0][0] for i in intersections if i[0][0] > 0]
        horizontal = [i[0][1] for i in intersections if i[0][1] > 0]
        # Get the min and max points
        min_v, max_v, = min(vertical), max(vertical)
        min_h, max_h = min(horizontal), max(horizontal)
        # Desired size of the image
        maxWidth = 1540
        maxHeight = 1000
        # Perspective crop process
        pts1 = np.float32([[min_v, min_h],[max_v, min_h],[min_v, max_h],[max_v, max_h]])
        pts2 = np.float32([[0, 0], [maxWidth, 0], [0, maxHeight], [maxWidth, maxHeight]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        croppedImage = cv2.warpPerspective(self.image, matrix, (maxWidth, maxHeight))
        # Save the result
        cv2.imwrite("houghResults/"+self.img_path.split("/")[1], cv2.cvtColor(croppedImage, cv2.COLOR_BGR2RGB))

    def run(self):
        self.read_image()
        self.image_preprocessing()
        self.hough_lines()
        self.crop_process()