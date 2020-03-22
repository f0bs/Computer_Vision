import tkinter as tk
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
import tkinter.ttk as ttk
import os
import math
import json
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw

import benchmark
import features

BUTTON_WIDTH = 14
SLIDER_LENGTH = 250

LEFT_IMAGE = 0
RIGHT_IMAGE = 1

GREEN = (0, 255, 0)
RED = (0, 0, 255)
AQUAMARINE = (212, 255, 127)

# The list of feature-types to be presented to the user
keypointClasses = [('Harris', features.HarrisKeypointDetector),
                   ('ORB', features.ORBKeypointDetector),
                   ('Dummy', features.DummyKeypointDetector)]

# The list of feature descriptors to be presented to the user
descriptorClasses = [('MOPS', features.MOPSFeatureDescriptor),
                     ('ORB', features.ORBFeatureDescriptor),
                     ('Simple', features.SimpleFeatureDescriptor),
                     ('Custom', features.CustomFeatureDescriptor)]

# The list of feature matching algorithms to be presented to the user
matcherClasses = [('Ratio Test', features.RatioFeatureMatcher),
                  ('SSD', features.SSDFeatureMatcher)]

# Supported filetypes
supportedFiletypes = [('JPEG Image', '*.jpg'), ('PNG Image', '*.png'),
 ('PPM Image', '*.ppm')]


def error(msg):
    tkMessageBox.showerror("Error", msg)


class CustomJSONEncoder(json.JSONEncoder):
    '''This class supports the serialization of JSON files containing
       OpenCV's feature and match objects as well as Numpy arrays.'''

    def __init__(self):
        super(CustomJSONEncoder, self).__init__(indent=True)

    def default(self, o):
        if hasattr(o, 'pt') and hasattr(o, 'size') and hasattr(o, 'angle') and \
           hasattr(o, 'response') and hasattr(o, 'octave') and \
           hasattr(o, 'class_id'):
            return {'__type__'  : 'cv2.KeyPoint',
                    'point'     : o.pt,
                    'size'      : o.size,
                    'angle'     : o.angle,
                    'response'  : o.response,
                    'octave'    : o.octave,
                    'class_id'  : o.class_id}

        elif hasattr(o, 'distance') and hasattr(o, 'trainIdx') and \
             hasattr(o, 'queryIdx') and hasattr(o, 'imgIdx'):
             return {'__type__' : 'cv2.DMatch',
                    'distance'  : o.distance,
                    'trainIdx'  : o.trainIdx,
                    'queryIdx'  : o.queryIdx,
                    'imgIdx'    : o.imgIdx}

        elif isinstance(o, np.ndarray):
            return {'__type__'  : 'numpy.ndarray',
                    '__shape__' : o.shape,
                    '__array__' : list(o.ravel())}
        else:
            json.JSONEncoder.default(self, o)


def customLoader(d):
    '''This function supports the deserialization of the custom types defined
       above.'''
    if '__type__' in d:
        if d['__type__'] == 'cv2.KeyPoint':
            k = cv2.KeyPoint()
            k.pt = (float(d['point'][0]), float(d['point'][1]))
            k.size = float(d['size'])
            k.angle = float(d['angle'])
            k.response = float(d['response'])
            k.octave = int(d['octave'])
            k.class_id = int(d['class_id'])
            return k
        elif d['__type__'] == 'cv2.DMatch':
            dm = cv2.DMatch()
            dm.distance = float(d['distance'])
            dm.trainIdx = int(d['trainIdx'])
            dm.queryIdx = int(d['queryIdx'])
            dm.imgIdx = int(d['imgIdx'])
            return dm
        elif d['__type__'] == 'numpy.ndarray':
            arr = np.array([float(x) for x in d['__array__']])
            arr.reshape(tuple([int(x) for x in d['__shape__']]))
            return arr
        else:
            return d
    else:
        return d


# Load a feature set from a file.
def load(filepath):
    return json.load(open(filepath, 'r'), object_hook=customLoader)


# Save a feature set to file.
def dump(filepath, obj):
    with open(filepath, 'w') as f:
        f.write(CustomJSONEncoder().encode(obj))


class ImageWidget(tk.Canvas):
    '''This class represents a Canvas on which OpenCV images can be drawn.
       The canvas handles shrinking of the image if the image is too big,
       as well as writing of the image to files. '''

    def __init__(self, parent):
        self.imageCanvas = tk.Canvas.__init__(self, parent)
        self.originalImage = None
        self.bind("<Configure>", self.redraw)

    def convertCVToTk(self, cvImage):
        height, width, _ = cvImage.shape
        if height == 0 or width == 0:
            return 0, 0, None
        img = Image.fromarray(cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB))
        return height, width, ImageTk.PhotoImage(img)

    def fitImageToCanvas(self, cvImage):
        height, width, _ = cvImage.shape
        if height == 0 or width == 0:
            return cvImage
        ratio = width / float(height)
        if self.winfo_height() < height:
            height = self.winfo_height()
            width = int(ratio * height)
        if self.winfo_width() < width:
            width = self.winfo_width()
            height = int(width / ratio)
        dest = cv2.resize(cvImage, (width, height),
            interpolation=cv2.INTER_LANCZOS4)
        return dest

    def drawCVImage(self, cvImage):
        self.originalImage = cvImage
        height, width, img = self.convertCVToTk(self.fitImageToCanvas(cvImage))
        if height == 0 or width == 0:
            return
        self.tkImage = img # prevent the image from being garbage collected
        self.delete("all")
        x = (self.winfo_width() - width) / 2.0
        y = (self.winfo_height() - height) / 2.0
        self.create_image(x, y, anchor=tk.NW, image=self.tkImage)

    def redraw(self, _):
        if self.originalImage is not None:
            self.drawCVImage(self.originalImage)

    def writeToFile(self, filename):
        if self.originalImage is not None:
            cv2.imwrite(filename, self.originalImage)


class BaseFrame(tk.Frame):
    '''The base frame inherited by all the tabs in the UI.'''

    def __init__(self, parent, root):
        tk.Frame.__init__(self, parent)
        self.grid(row=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.root = root

        # UI elements which appear in all frames
        # We don't specify the position here, because it varies among
        # different frames
        self.status = tk.Label(self, text='Welcome to Features UI')

        self.thresholdLabel = tk.Label(self, text='Threshold (10^x):')
        self.thresholdSlider = tk.Scale(self, from_=-4, to=0, resolution=0.1,
            orient=tk.HORIZONTAL, length=SLIDER_LENGTH)
        self.thresholdSlider.set(-2)

        self.keypointLabel = tk.Label(self, text='Keypoint Type:')
        self.keypointTypeVar = tk.StringVar(self)
        self.keypointTypeVar.set(keypointClasses[0][0])
        self.keypointOptions = tk.OptionMenu(*((self, self.keypointTypeVar)\
         + tuple(x[0] for x in keypointClasses)))
        self.keypointOptions.configure(width=BUTTON_WIDTH)

        self.descriptorLabel = tk.Label(self, text='Descriptor Type:')
        self.descriptorTypeVar = tk.StringVar(self)
        self.descriptorTypeVar.set(descriptorClasses[0][0])
        self.descriptorOptions = tk.OptionMenu(*((self, self.descriptorTypeVar)\
         + tuple(x[0] for x in descriptorClasses)))
        self.descriptorOptions.configure(width=BUTTON_WIDTH)

        self.matcherLabel = tk.Label(self, text='Matcher Type:')
        self.matcherTypeVar = tk.StringVar(self)
        self.matcherTypeVar.set(matcherClasses[0][0])
        self.matcherOptions = tk.OptionMenu(*((self, self.matcherTypeVar) +
            tuple(x[0] for x in matcherClasses)))
        self.matcherOptions.configure(width=BUTTON_WIDTH)

        self.imageCanvas = ImageWidget(self)

        for i in range(6):
            self.grid_columnconfigure(i, weight=1)

        self.grid_rowconfigure(3, weight=1)

    def getSelectedDetector(self):
        index = [x[0] for x in keypointClasses].index(
            self.keypointTypeVar.get())
        keypointClass = keypointClasses[index][1]
        return keypointClass()

    def getSelectedDescriptor(self):
        index = [x[0] for x in descriptorClasses].index(
            self.descriptorTypeVar.get())
        descriptorClass = descriptorClasses[index][1]
        return descriptorClass()

    def getSelectedMatcher(self):
        index = [x[0] for x in matcherClasses].index(self.matcherTypeVar.get())
        matcherClass = matcherClasses[index][1]
        return matcherClass()

    def getSelectedKpThreshold(self):
        return 10**float(self.thresholdSlider.get())

    def setStatus(self, text):
        self.status.configure(text=text)

    def thresholdKeyPoints(self, keypoints, threshold):
        return [kp for kp in keypoints if kp.response >= threshold]


class KeypointDetectionFrame(BaseFrame):
    def __init__(self, parent, root):
        BaseFrame.__init__(self, parent, root)

        self.loadImageButton = tk.Button(self, text='Load Image',
            command=self.loadImage, width=BUTTON_WIDTH)
        self.loadImageButton.grid(row=0, column=0, sticky=tk.W+tk.E)

        self.clearKeypointsButton = tk.Button(self, text='Clear Keypoints',
            command=self.reloadImage, width=BUTTON_WIDTH)
        self.clearKeypointsButton.grid(row=0, column=1, sticky=tk.W+tk.E)

        self.screenshotButton = tk.Button(self, text='Screenshot',
            command=self.screenshot, width=BUTTON_WIDTH)
        self.screenshotButton.grid(row=0, column=4, sticky=tk.W+tk.E)

        self.computeButton = tk.Button(self, text='Compute',
            command=self.computeKeypoints, width=BUTTON_WIDTH)
        self.computeButton.grid(row=0, column=5, sticky=tk.W+tk.E)

        self.keypointLabel.grid(row=1, column=0, sticky=tk.W)
        self.keypointOptions.grid(row=1, column=1, sticky=tk.W+tk.E)
        self.keypointTypeVar.trace("w", self.computeKeypoints)

        self.thresholdLabel.grid(row=1, column=3, sticky=tk.W)
        self.thresholdSlider.grid(row=1, column=4, columnspan=2,
            sticky=tk.W+tk.E)
        self.thresholdSlider.configure(command=self.drawKeypoints)

        self.imageCanvas.grid(row=3, columnspan=6, sticky=tk.N+tk.S+tk.E+tk.W)

        self.status.grid(row=4, columnspan=6, sticky=tk.S)

        self.image = None
        self.keypoints = None

    def loadImage(self):
        filename = tkFileDialog.askopenfilename(parent=self.root,
            filetypes=supportedFiletypes)
        if filename and os.path.isfile(filename):
            self.image = cv2.imread(filename)
            self.imageCanvas.drawCVImage(self.image)
            self.setStatus('Loaded ' + filename)

    def computeKeypoints(self, *args):
        if self.image is not None:
            self.reloadImage()
            detector = self.getSelectedDetector()
            self.keypoints = detector.detectKeypoints(self.image)
            self.drawKeypoints()
        else:
            error('Load image before computing keypoints!')

    def drawKeypoints(self, val=None):
        threshold = self.getSelectedKpThreshold()
        if self.image is not None and self.keypoints is not None:
            kps = self.thresholdKeyPoints(self.keypoints, threshold)
            # img = cv2.drawKeypoints(self.image, kps,
            #     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            #     color=GREEN)
            img = self.image.copy()
            for marker in kps:
                img = cv2.drawMarker(img, tuple(int(i) for i in marker.pt), color=GREEN)
            self.imageCanvas.drawCVImage(img)
            self.setStatus('Found ' + str(len(kps)) +
                ' keypoints')

    def reloadImage(self):
        if self.image is not None:
            self.keypoints = None
            self.imageCanvas.drawCVImage(self.image)

    def screenshot(self):
        if self.image is not None:
            filename = tkFileDialog.asksaveasfilename(parent=self.root,
                filetypes=supportedFiletypes, defaultextension=".png")
            if filename:
                self.imageCanvas.writeToFile(filename)
                self.setStatus('Saved screenshot to ' + filename)
        else:
            error('Load image before taking a screenshot!')


class FeatureMatchingFrame(BaseFrame):
    def __init__(self, parent, root):
        BaseFrame.__init__(self, parent, root)

        self.leftRightSelection = tk.IntVar()

        self.loadLeftImageButton = tk.Button(self, text='Load Left Image',
            command=self.loadLeftImage, width=BUTTON_WIDTH)
        self.loadLeftImageButton.grid(row=0, column=0, sticky=tk.W+tk.E)

        self.loadRightImageButton = tk.Button(self, text='Load Right Image',
            command=self.loadRightImage, width=BUTTON_WIDTH)
        self.loadRightImageButton.grid(row=0, column=1, sticky=tk.W+tk.E)

        self.clearMatchingsButton = tk.Button(self, text='Clear Matchings',
            command=self.clearMatchings, width=BUTTON_WIDTH)
        self.clearMatchingsButton.grid(row=0, column=3, sticky=tk.W+tk.E)

        self.screenshotButton = tk.Button(self, text='Screenshot',
            command=self.screenshot, width=BUTTON_WIDTH)
        self.screenshotButton.grid(row=0, column=4, sticky=tk.W+tk.E)

        self.computeMatchesButton = tk.Button(self, text='Compute',
            command=self.computeMatchesClick, width=BUTTON_WIDTH)
        self.computeMatchesButton.grid(row=0, column=5, sticky=tk.W+tk.E)

        self.keypointLabel.grid(row=1, column=0, sticky=tk.W)
        self.keypointOptions.grid(row=1, column=1, sticky=tk.W+tk.E)
        self.keypointTypeVar.trace("w", self.computeMatches)

        self.descriptorLabel.grid(row=1, column=2, sticky=tk.W)
        self.descriptorOptions.grid(row=1, column=3, sticky=tk.W+tk.E)
        self.descriptorTypeVar.trace("w", self.computeMatches)

        self.matcherLabel.grid(row=1, column=4, sticky=tk.W)
        self.matcherOptions.grid(row=1, column=5, sticky=tk.W+tk.E)
        self.matcherTypeVar.trace("w", self.computeMatches)

        self.thresholdLabel.grid(row=2, column=0, sticky=tk.W)
        self.thresholdSlider.grid(row=2, column=1, columnspan=2,
            sticky=tk.W+tk.E)
        self.thresholdSlider.bind("<ButtonRelease-1>", self.thresholdChanged)

        tk.Label(self, text='Percent Matches:').grid(row=2, column=3,
            sticky=tk.W)

        self.percentMatches = tk.Scale(self, from_=0, to=100, resolution=0.1,
            orient=tk.HORIZONTAL, length=SLIDER_LENGTH,
            command=self.updateMatchCount)
        self.percentMatches.grid(row=2, column=4, columnspan=2,
            sticky=tk.W+tk.E)
        self.percentMatches.set(5)

        self.imageCanvas.grid(row=3, columnspan=6, sticky=tk.N+tk.S+tk.E+tk.W)

        self.status.grid(row=4, columnspan=6)

        self.resetState()
        self.image = [None, None]

    def resetState(self):
        self.keypoints = [None, None]
        self.descriptors = [None, None]
        self.matches = None

    def thresholdChanged(self, val):
        self.computeMatches()

    def clearMatchings(self):
        if self.image[0] is not None and self.image[1] is not None:
            self.resetState()
            self.drawImages()
            self.setStatus('Cleared matchings')

    def loadLeftImage(self):
        self.loadImage(0)

    def loadRightImage(self):
        self.loadImage(1)

    def loadImage(self, index):
        filename = tkFileDialog.askopenfilename(parent=self.root,
        filetypes=supportedFiletypes)
        if filename and os.path.isfile(filename):
            self.image[index] = cv2.imread(filename)
            self.setStatus('Loaded ' + filename)
            self.resetState()
            self.drawImages()

    def screenshot(self):
        if self.image[0] is not None or self.image[1] is not None:
            filename = tkFileDialog.asksaveasfilename(parent=self.root,
                filetypes=supportedFiletypes, defaultextension=".png")
            if filename:
                self.imageCanvas.writeToFile(filename)
                self.setStatus('Saved screenshot to ' + filename)
        else:
            error('Load image before taking a screenshot!')

    def updateMatchCount(self, percent):
        if self.matches is not None:
            matchCount = int(float(percent) * len(self.matches) / 100)
            matches = self.matches[:matchCount]
            if len(matches) == 0:
                self.drawImages()
            else:
                matchImage = self.drawMatches(self.image[0],
                    self.thresholdedKeypoints[0], self.image[1],
                    self.thresholdedKeypoints[1], matches)
                self.imageCanvas.drawCVImage(matchImage)

    def thresholdAndMatch(self):
        threshold = self.getSelectedKpThreshold()

        self.setStatus('Computing descriptors')

        descriptor = self.getSelectedDescriptor()
        self.thresholdedKeypoints = [None] * len(self.image)

        for i in range(2):
            self.thresholdedKeypoints[i] = self.thresholdKeyPoints(
                self.keypoints[i], self.getSelectedKpThreshold())
            self.descriptors[i] = descriptor.describeFeatures(self.image[i],
                self.thresholdedKeypoints[i])

        self.setStatus('Finding matches')

        matcher = self.getSelectedMatcher()
        matches = matcher.matchFeatures(self.descriptors[0],self.descriptors[1])
        self.matches = sorted(matches, key = lambda x : x.distance)

    def concatImages(self, imgs):
        # Skip Nones
        imgs = [img for img in imgs if img is not None]
        maxh = max([img.shape[0] for img in imgs]) if imgs else 0
        sumw = sum([img.shape[1] for img in imgs]) if imgs else 0
        vis = np.zeros((maxh, sumw, 3), np.uint8)
        vis.fill(255)
        accumw = 0
        for img in imgs:
            h, w = img.shape[:2]
            vis[:h, accumw:accumw+w, :] = img
            accumw += w

        return vis

    def drawImages(self):
        result = self.concatImages(self.image)
        self.imageCanvas.drawCVImage(result)

    def computeMatchesClick(self):
        if self.image[0] is not None and self.image[1] is not None:
            self.computeMatches()
        else:
            error('Load images before computing matches!')

    def computeMatches(self, *args):
        if self.image[0] is not None and self.image[1] is not None:
            self.setStatus('Finding keypoints')

            threshold = self.getSelectedKpThreshold()
            detector = self.getSelectedDetector()
            for i in range(2):
                if self.keypoints[i] is None:
                    self.keypoints[i] = detector.detectKeypoints(self.image[i])

            self.thresholdAndMatch()

            self.setStatus('Found {} matches'.format(len(self.matches)))

            self.updateMatchCount(self.percentMatches.get())

    def drawMatches(self, img1, kp1, img2, kp2, matches):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        vis = self.concatImages([img1, img2])

        kp_pairs = [[kp1[m.queryIdx], kp2[m.trainIdx]] for m in matches]
        status = np.ones(len(kp_pairs), np.bool_)
        p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
        p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)

        green = (0, 255, 0)
        red = (0, 0, 255)
        white = (255, 255, 255)
        kp_color = (51, 103, 236)
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                cv2.circle(vis, (x1, y1), 5, GREEN, 2)
                cv2.circle(vis, (x2, y2), 5, GREEN, 2)
            else:
                r = 5
                thickness = 6
                cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), RED, thickness)
                cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), RED, thickness)
                cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), RED, thickness)
                cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), RED, thickness)
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), AQUAMARINE)

        return vis


class BenchmarkFrame(BaseFrame):
    def __init__(self, parent, root):
        BaseFrame.__init__(self, parent, root)

        self.thresholdLabel.grid(row=0, column=0, sticky=tk.W)
        self.thresholdSlider.grid(row=0, column=1, columnspan=2,
            sticky=tk.W+tk.E)
        self.thresholdSlider.bind("<ButtonRelease-1>", self.runBenchmark)

        self.screenshotButton = tk.Button(self, text='Screenshot',
            command=self.screenshot, width=BUTTON_WIDTH)
        self.screenshotButton.grid(row=0, column=4, sticky=tk.W+tk.E)

        self.runBenchmarkButton = tk.Button(self, text='Run benchmark',
            command=self.runBenchmarkClick, width=BUTTON_WIDTH)
        self.runBenchmarkButton.grid(row=0, column=5, sticky=tk.W+tk.E)

        self.keypointLabel.grid(row=1, column=0, sticky=tk.W)
        self.descriptorLabel.grid(row=1, column=2, sticky=tk.W)
        self.matcherLabel.grid(row=1, column=4, sticky=tk.W)

        self.keypointOptions.grid(row=1, column=1, sticky=tk.W+tk.E)
        self.keypointTypeVar.trace("w", self.runBenchmark)
        self.descriptorOptions.grid(row=1, column=3, sticky=tk.W+tk.E)
        self.descriptorTypeVar.trace("w", self.runBenchmark)
        self.matcherOptions.grid(row=1, column=5, sticky=tk.W+tk.E)
        self.matcherTypeVar.trace("w", self.runBenchmark)

        self.imageCanvas.grid(row=3, columnspan=6, sticky=tk.N+tk.S+tk.E+tk.W)

        self.status.grid(row=4, columnspan=6)

        self.currentDirectory = None

    def runBenchmarkClick(self):
        dirpath = tkFileDialog.askdirectory(parent=self.root)
        if not dirpath:
            return
        self.currentDirectory = dirpath
        self.runBenchmark()

        self.currentDirectory = dirpath

    def screenshot(self):
        if self.roc_img is not None:
            filename = tkFileDialog.asksaveasfilename(parent=self.root,
                filetypes=supportedFiletypes, defaultextension=".png")
            if filename:
                self.imageCanvas.writeToFile(filename)
                self.setStatus('Saved screenshot to ' + filename)

    def runBenchmark(self, *args):
        if self.currentDirectory:
            detector = self.getSelectedDetector()
            descriptor = self.getSelectedDescriptor()
            matcher = self.getSelectedMatcher()
            kpThreshold = self.getSelectedKpThreshold()
            self.setStatus('Benchmarking...Please wait.')
            matchThreshold = 5
            ds, aucs, roc_img = benchmark.benchmark_dir(self.currentDirectory,
                detector, descriptor, matcher, kpThreshold, matchThreshold)
            self.roc_img = roc_img
            self.imageCanvas.drawCVImage(roc_img)
            text = 'Average distance between true and actual matches: {}; \
                Average AUC: {}'.format(np.mean(ds), np.mean(aucs))
            self.setStatus(text)

class FeaturesUIFrame(tk.Frame):
    def __init__(self, parent, root):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.root = root
        self.notebook = ttk.Notebook(self.parent)
        self.keypointDetectionFrame = KeypointDetectionFrame(self.notebook, root)
        self.featureMatchingFrame = FeatureMatchingFrame(self.notebook, root)
        self.benchmarkFrame = BenchmarkFrame(self.notebook, root)
        self.notebook.add(self.keypointDetectionFrame, text='Keypoint Detection')
        self.notebook.add(self.featureMatchingFrame, text='Feature Matching')
        self.notebook.add(self.benchmarkFrame, text='Benchmark')
        self.notebook.grid(row=0, sticky=tk.N+tk.S+tk.E+tk.W)

    def CloseWindow(self):
        self.root.quit()


if __name__ == '__main__':
    root = tk.Tk()
    app = FeaturesUIFrame(root, root)
    root.title('Cornell CS 4670 - Feature Detection Project')
    # Put the window on top of the other windows
    #root.attributes('-fullscreen', True)
    w, h = root.winfo_screenwidth(), root.winfo_screenheight() - 50
    root.geometry("%dx%d+0+0" % (w, h))
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    #root.wm_attributes('-topmost', 1)
    root.mainloop()
