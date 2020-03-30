# CS5670 Computer Vision Projects and Artefacts


## Project 3: Creating Panorama Pictures with Autostitch

### Introduction

Building our own Autostitch system to combine overlapping photographs into a single panorama

* Feature matching using ORB feature detector (opencv)

* Using **RANSAC** to align the photographs

* Blend the images with alpha blending (feathering)

* Correcting for drift and create 360° panoramas 

Click [here](http://www.cs.cornell.edu/courses/cs5670/2020sp/projects/pa3/index.html) to view projects introduction. 


### Steps

1. Take pictures with a camera / phone

2. Warp to spherical coordinates

3. Extract features

4. Match features

5. Align neighboring pairs using RANSAC

6. Write out list of neighboring translations

7. Correct for drift

8. Read in images and blend them

9. Crop the result and import into a viewer
  
### Structure

| Name         | Function                                        |
| ------------ | ----------------------------------------------- |
| /resources   | Images used to create panoramas                 |
| warp.py      | Warp images into spherical coordinates.         |
| alignment.py | Compute the alignment of image pairs.           |
| blend.py     | Stitch and blend the aligned images.            |
| test.py      | Test cases provided to test and debug our code  |
| gui.py       | Gui provided to create panorama                 |

#### Libraries used

* Python version: 3.7.4
* cv2
* NumPy
* SciPy


### Example outputs

#### Yosemite

##### Input
| ![](/Project3_Panorama_Autostitch/resources/yosemite/panorama/yosemite1.jpg) | ![](/Project3_Panorama_Autostitch/resources/yosemite/panorama/yosemite2.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](/Project3_Panorama_Autostitch/resources/yosemite/panorama/yosemite3.jpg) | ![](/Project3_Panorama_Autostitch/resources/yosemite/panorama/yosemite4.jpg) |

##### Panorama

![](/Project3_Panorama_Autostitch/resources/yosemite/yosemite_pano_homography_blendwidth50.png)
