#!/usr/bin/python
# handle = vot.VOT("rectangle")
import vot
import sys
import time
import cv2
import numpy
import collections
import flow
import simulator
import ORF
from ncc import NCCTracker
import medianFlow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Rectangle
import vot
import sys
import time
import cv2
import numpy
import collections
from camShift import camShift


sim = True
if sim:
    #handle = simulator.simulator("/home/boka/arp/david/")
    handle = simulator.simulator("/home/boka/arp/vot-toolkit/workspace/sequences/cup/")
    #handle = simulator.simulator("/home/boka/arp/vot-toolkit/workspace/sequences/woman/")
    #handle = simulator.simulator("/home/boka/arp/vot-toolkit/workspace/sequences/juice/")
    #handle = simulator.simulator("/home/boka/arp/vot-toolkit/workspace/sequences/jump/")
else:
    handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
print("prvo")
if not imagefile:
    sys.exit(0)

# image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(imagefile, cv2.IMREAD_COLOR)
print(imagefile)
tracker = NCCTracker(image, selection)
tracker_flow = medianFlow.flow(image, selection)
tracker_OT = ORF.flow(image, selection)
tracker_mean = camShift(image, selection)
print("do tukej")
if sim:
    plt.ion()
    plt.figure()
while True:
    imagefile = handle.frame()
    if not imagefile:
        print("break")
        break

    image = cv2.imread(imagefile, cv2.IMREAD_COLOR)
    #print("zac")
    region = tracker.track(image)
    #print("ncc")
    [conf, regionOrg] = tracker_OT.track(image)
    #print("orf")
    region_flow = tracker_flow.track(image)
    region_mean = tracker_mean.track(image)
    #print("shft")
    if (abs(region.x - regionOrg.x) / float(region.width) < 0.05 and abs(region.y - regionOrg.y) / float(region.height) < 0.05) or (
            abs(region_flow.x - regionOrg.x) / float(region.width) < 0.05 and abs(
                    region_flow.y - regionOrg.y) / float(region.height) < 0.05):
        print("updatalo bo")
        tracker_OT.updateTree(image)
        print("updatalo je tree")

    if conf > 0.70 and (abs(region_flow.x - regionOrg.x) / float(region.width) > 0.3 or abs(
                region_flow.y - regionOrg.y) / float(region.height) > 0.3):
        tracker_flow.set_position(tracker_OT.position)
        #region_flow = regionOrg
        print("popravil polozaj")
    #print("popravki")
    handle.report(region_flow)
    if sim:
        plt.clf()
        a = plt.imshow(image)
        currentAxis = plt.gca()
        currentAxis.add_patch(
            Rectangle((region.x, region.y), region.width, region.height, fill=None, alpha=1, color='yellow'))
        currentAxis.add_patch(
            Rectangle((regionOrg.x, regionOrg.y), regionOrg.width, regionOrg.height, fill=None, alpha=1, color='green'))
        currentAxis.add_patch(
            Rectangle((region_flow.x, region_flow.y), region_flow.width, region_flow.height, fill=None, alpha=1,
                      color='red'))
        currentAxis.add_patch(
            Rectangle((region_mean.x, region_mean.y), region_mean.width, region_mean.height, fill=None, alpha=1,
                      color='cyan'))
        plt.draw()
        time.sleep(0.1)

if sim:
    plt.show()
