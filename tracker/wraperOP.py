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
import tensor
from sklearn.metrics.pairwise import euclidean_distances
from multiprocessing import Process, Pipe



sim = False
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
root = image[selection.y:selection.y + selection.height,selection.x:selection.x + selection.width]
print("do tukej")
if sim:
    plt.ion()
    plt.figure()

stevec = 0
while True:
    plt.clf()
    imagefile = handle.frame()
    if not imagefile:
        print("break")
        break

    image = cv2.imread(imagefile, cv2.IMREAD_COLOR)
    region = tracker.track(image)
    [conf, regionOrg] = tracker_OT.track(image)
    region_flow = tracker_flow.track(image)
    region_mean = tracker_mean.track(image)

    regions = [region, region_flow, regionOrg, region_mean]
    #print("shft")
    if (abs(region.x - regionOrg.x) / float(region.width) < 0.05 and abs(region.y - regionOrg.y) / float(region.height) < 0.05) or (
            abs(region_flow.x - regionOrg.x) / float(region.width) < 0.05 and abs(
                    region_flow.y - regionOrg.y) / float(region.height) < 0.05):
        tracker_OT.updateTree(image)


    if conf > 0.70 and (abs(region_flow.x - regionOrg.x) / float(region.width) > 0.3 or abs(
                region_flow.y - regionOrg.y) / float(region.height) > 0.3):
        print("popravil bi mogoce")
        if not ((abs(region_flow.x - region_mean.x) / float(region_flow.width) < 0.3 and abs(region_mean.y - region_flow.y) / float(region_flow.height) < 0.3) or (
            abs(region_flow.x - region_mean.x) / float(region_mean.width) < 0.3 and abs(
                    region_flow.y - region_mean.y) / float(region_mean.height) < 0.3)):
            tracker_flow.set_position(tracker_OT.position)
            tracker_mean.set_position(tracker_OT.position)
            #region_flow = regionOrg
            print("popravil polozaj")


    if stevec == 0:
        images = [root]
        for r in regions:
            print(r.x)
            print(r.height)
            cut = image[r.y:r.y + r.height,r.x:r.x+r.width]
            images += [cut]
        print(images)
        proces, proces2 = Pipe()
        p = Process(target=tensor.get_closest, args=(proces2, images))
        p.start()

    stevec+=1
    if stevec == 8:
        rec = proces.recv()[0]
        stevec = 0
        p.join()
        if rec == 1:
            tracker.set_region(regions[rec])
            tracker_flow.set_region(regions[rec])
            tracker_mean.set_region(regions[rec])
            tracker_OT.set_region(regions[rec])

            region = regions[rec]
            region_flow = regions[rec]
            region_mean = regions[rec]
            regionOrg  = regions[rec]


    handle.report(region_flow)
    if sim:

        #plt.figure(2)
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

