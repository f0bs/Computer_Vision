import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
import cv2
from PIL import Image

import features
from features import *


def plot_2D_arrays(title, arrs, xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[]):

    for arr in arrs:
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError('The array should be 2D and the second dimension should be 2!')

        plt.plot(arr[:, 0], arr[:, 1])

    plt.title(title)
    plt.xlabel(xlabel)
    if xinterval:
        plt.xlim(xinterval)

    plt.ylabel(ylabel)
    if yinterval:
        plt.ylim(yinterval)

    if line_names:
        plt.legend(line_names, loc='best')

    pylab.savefig('__temp.png', bbox_inches='tight')
    plt.clf()

    img = cv2.imread('__temp.png')
    os.remove('__temp.png')
    return img


def plot_2D_array(title, arr, xlabel='', xinterval=None, ylabel='', yinterval=None):
    return plot_2D_arrays(title, [arr], xlabel, xinterval, ylabel, yinterval, line_names=[])


def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf


def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.fromarray(buf, mode="RGBA").convert("RGB")



def addROCData(f1, f2, matches, h, threshold):
    isMatch = []
    maxD = 0

    dists = []
    for m in matches:
        id1 = m.queryIdx
        id2 = m.trainIdx

        ptOld = np.array(f2[id2].pt)
        ptNew = FeatureMatcher.applyHomography(f1[id1].pt, h)

        #Ignore unmatched points.  There might be a better way to
        #handle this.
        # Euclidean distance
        d = np.linalg.norm(ptNew - ptOld)
        dists.append(d)
        if d <= threshold:
            isMatch.append(1)
        else:
            isMatch.append(0)

        if m.distance > maxD:
            maxD = m.distance

    #plt.hist(dists)
    #plt.show()

    return isMatch, maxD


def computeROCCurve(matches, isMatch, thresholds):
    dataPoints = []

    for threshold in thresholds:
        tp = 0
        actualCorrect = 0
        fp = 0
        actualError = 0
        total = 0

        for j in range(len(matches)):
            if isMatch[j]:
                actualCorrect += 1
                if matches[j].distance < threshold:
                    tp += 1
            else:
                actualError += 1
                if matches[j].distance < threshold:
                    fp += 1

            total += 1

        trueRate = (float(tp) / actualCorrect) if actualCorrect != 0 else 0
        falseRate = (float(fp) / actualError) if actualError != 0 else 0

        dataPoints.append((falseRate, trueRate))

    return dataPoints


def computeAUC(results):
    auc = 0

    for i in range(1, len(results)):
        falseRate, trueRate = results[i]
        falseRatePrev, trueRatePrev = results[i - 1]
        xdiff = falseRate - falseRatePrev
        ydiff = trueRate - trueRatePrev
        auc += xdiff * trueRatePrev + xdiff * ydiff / 2

    return auc


def load_homography(filename):
    with open(filename) as f:
        content = f.readlines()
        homography = [float(n) for c in content for n in c.split()]
    return homography


def benchmark_dir(dirpath, keypointDetector, featureDescriptor, featureMatcher,
    kpThreshold, matchThreshold):
    image_pattern = '^.+(\\d+)(?:(?:\\.ppm)|(?:\\.png)|(?:\\.jpg))$'
    homography_pattern = '^H(\\d+)to(\\d+)p$'
    filenames = os.listdir(dirpath)

    origImageName = ''
    trafoImageNames = {}
    homographyNames = {}

    for fn in filenames:
        match = re.match(image_pattern, fn)
        if match:
            imgNum = int(match.group(1))
            if imgNum == 1:
                origImageName = fn
            else:
                trafoImageNames[imgNum] = fn

        match = re.match(homography_pattern, fn)
        if match:
            fromImgNum = int(match.group(1))
            toImgNum = int(match.group(2))
            if fromImgNum != 1:
                raise ValueError('Homography file should transform image #1')

            homographyNames[toImgNum] = fn

    sortedkeys = sorted(trafoImageNames)
    #print 'Original image name: {}'.format(origImageName)
    #print 'Trasformed image names: {}'.format(trafoImageNames)
    #print 'Homography file names: {}'.format(homographyNames)
    origImage = cv2.imread(os.path.join(dirpath, origImageName))

    trafoImages = []
    homographies = []
    for imgNum in sortedkeys:
        trafoImage = cv2.imread(os.path.join(dirpath, trafoImageNames[imgNum]))
        h = load_homography(os.path.join(dirpath, homographyNames[imgNum]))

        trafoImages.append(trafoImage)
        homographies.append(h)

    return benchmark(origImage, trafoImages, homographies,
              keypointDetector, featureDescriptor,
              featureMatcher, kpThreshold, matchThreshold)


def benchmark(origImage, trafoImages, homographies,
              keypointDetector, featureDescriptor,
              featureMatcher, kpThreshold, matchThreshold):
    '''
        Input:
            origImage -- The original image which is transformed
            trafoImages -- List of images, transformed from origImage
                using homographies
            homographies -- List of homographies (numpy arrays),
                the length should be equal to the length of trafoImages
            keypointDetector -- The selected keypoint detector algorithm
            featureDescriptor -- The selected feature descriptor algorithm
            featureMatcher -- The selected feature matcher algorithm
            kpThreshold -- The threshold used for keypoint detection
            matchThreshold -- The threshold used to determine if a match is valid
    '''
    assert len(trafoImages) == len(homographies)
    okps = keypointDetector.detectKeypoints(origImage)
    okps = [kp for kp in okps if kp.response >= kpThreshold]
    odesc = featureDescriptor.describeFeatures(origImage, okps)

    ds = []
    aucs = []
    data_point_list = []
    line_legends = []
    # go through each transformed image and perform feature matching
    for i, timg in enumerate(trafoImages):
        #print 'Matching image 1 with image {}'.format(i+2)
        tkps = keypointDetector.detectKeypoints(timg)
        tkps = [kp for kp in tkps if kp.response >= kpThreshold]
        tdesc = featureDescriptor.describeFeatures(timg, tkps)
        matches = featureMatcher.matchFeatures(odesc, tdesc)
        matches = sorted(matches, key = lambda x:x.distance)

        d = features.FeatureMatcher.evaluateMatch(
            okps, tkps, matches,
            homographies[i])
        ds.append(d)

        isMatch, maxD = addROCData(
            okps, tkps, matches,
            homographies[i], matchThreshold)

        thresholdList = np.linspace(0.0, maxD+1, num=500)
        dataPoints = computeROCCurve(matches, isMatch, thresholdList)
        auc = computeAUC(dataPoints)
        aucs.append(auc)
        data_point_list.append(np.array(dataPoints))
        line_legends.append('1 vs {}'.format(i+2))

    roc_img = plot_2D_arrays(
        'All plots', data_point_list, xlabel='False rate',
        ylabel = 'True rate', line_names=line_legends)

    return ds, aucs, roc_img

