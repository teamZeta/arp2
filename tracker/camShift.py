import vot
import sys
import time
import cv2
import numpy
import collections
import simulator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Rectangle

class camShift(object):

    def __init__(self, image, region):
        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)

        roi = image[int(top):int(bottom), int(left):int(right), :]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 10., 32.)), np.array((180., 255., 255.)))
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.bb = [left, top, region.width, region.height]


    def set_position(self, position):
        self.position = (position[0], position[1])
        self.bb = [position[0] - self.size[0] / 2, position[1] - self.size[1] / 2, self.size[0],
                   self.size[1]]

    def track(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        ret, track_window = cv2.meanShift(dst, (self.bb[0], self.bb[1], self.bb[2], self.bb[3]), self.term_crit)
        self.position = (track_window[0] + int(track_window[2]/2), track_window[1] + int(track_window[3]/2))
        #self.size = (int(track_window[2]), int(track_window[3]) )
        self.bb = track_window
        print(track_window)
        return vot.Rectangle(track_window[0], track_window[1], track_window[2], track_window[3])

