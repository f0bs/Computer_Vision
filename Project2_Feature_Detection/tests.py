import numpy as np
import sys, os, imp
import cv2
import transformations
import features
import traceback

from PIL import Image

# Saving and loading cv2 points
def pickle_cv2(arr):
    index = []
    for point in arr:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        index.append(temp)
    return np.array(index)

def unpickle_cv2(arr):
    index = []
    for point in arr:
        temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        index.append(temp)
    return np.array(index)

# Functions for testing elementwise correctness
def compare_array(arr1, arr2):
    return np.allclose(arr1,arr2,rtol=1e-3,atol=1e-5)

def compare_cv2_points(pnt1, pnt2):
    if not np.isclose(pnt1.pt[0],pnt2.pt[0],rtol=1e-3,atol=1e-5): return False
    if not np.isclose(pnt1.pt[1],pnt2.pt[1],rtol=1e-3,atol=1e-5): return False
    if not np.isclose(pnt1.angle,pnt2.angle,rtol=1e-3,atol=1e-5): return False
    if not np.isclose(pnt1.response,pnt2.response,rtol=1e-3,atol=1e-5): return False
    return True

# Testing function
def try_this(todo, run, truth, compare, *args, **kargs):
    '''
    Run a function, test the output with compare, and print and error if it doesn't work
    @arg todo (int or str): The Todo number
    @arg run (func): The function to run
    @arg truth (any): The correct output of the function 
    @arg compare (func->bool): Compares the output of the `run` function to truth and provides a boolean if correct
    @arg *args (any): Any arguments that should be passed to `run`
    @arg **kargs (any): Any kargs that should be passed to compare

    @return (int): The amount of things that failed
    '''
    print('Starting test for TODO {}'.format(todo))
    failed = 0
    try:
        output = run(*args)
        print("ok")
    except Exception as e:
        traceback.print_exc()
        print("TODO {} threw an exception, see exception above".format(todo))
        return
    if type(output) is list or type(output) is tuple:
        for i in range(len(output)):
            if not compare(output[i], truth[i], **kargs):
                print("TODO {} doesn't pass test: {}".format(todo, i))
                failed+=1
    else:
        if not compare(output, truth, **kargs):
            print("TODO {} doesn't pass test".format(todo))
            failed+=1
    return failed

HKD = features.HarrisKeypointDetector()
SFD = features.SimpleFeatureDescriptor()
MFD = features.MOPSFeatureDescriptor()
SSDFM = features.SSDFeatureMatcher()

image = np.array(Image.open('resources/triangle1.jpg'))
grayImage = cv2.cvtColor(image.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
def compute_and_save():
    (a,b) = HKD.computeHarrisValues(grayImage) # Todo1
    c = HKD.computeLocalMaxima(a) # Todo2
    d = HKD.detectKeypoints(image) # Todo3
    e = SFD.describeFeatures(image, d) # Todo 4
    f = MFD.describeFeatures(image, d) # Todo 5,6
    # No test for Todo 7 or 8
    d_proc = pickle_cv2(d)
    np.savez('resources/arrays',a=a,b=b,c=c,d_proc=d_proc,e=e,f=f)
# Uncomment next line to overwrite test data (not recommended)
#compute_and_save()


'''
Load in the numpy arrays which hold results for triangle1.jpg.

These arrays can be accessed using loaded['<letter>']. For example, the
correct output for test 2 is 'c', so to see the correct output for test
2 you can inspect loaded['c'].  Important note: NumPy does not print
the entire array if it is very large --- you must print smaller pieces
(e.g., print repr(loaded['c'][0])).

If your tests fail you should inspect why it failed. In particular,
pay attention to the tolerances used by this testing script. It is
possible that your answer is correct but it barely falls outside the
tolerance range.

This is not the script used by the autograder. 
'''
loaded = np.load('resources/arrays.npz')
d = unpickle_cv2(loaded['d_proc'])

try_this(1, HKD.computeHarrisValues, [loaded['a'],loaded['b']], compare_array, grayImage)

# patch HKD so future tests won't fail because the last test failed
class HKD2(features.HarrisKeypointDetector):
  def computeHarrisValues(self,image):
    return loaded['a'],loaded['b']
HKD=HKD2()

try_this(2, HKD.computeLocalMaxima, loaded['c'], compare_array, loaded['a'])

# patch HKD so future tests won't fail because the last test failed
class HKD3(HKD2):
  def computeLocalMaxima(self,image):
    return loaded['c']
HKD=HKD3()

try_this(3, HKD.detectKeypoints, d, compare_cv2_points, image)

try_this(4, SFD.describeFeatures, loaded['e'], compare_array, image, d)

try_this('5 and/or 6', MFD.describeFeatures, loaded['f'], compare_array, image, d)

