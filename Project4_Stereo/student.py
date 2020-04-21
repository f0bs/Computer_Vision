import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    # Create I a vector of N images from the list images by reshaping such
    # that each image is an array of size 1980*1080*c with c=3 for RGB or 1 for greyscale
    images_shape = np.shape(images)
    I = np.array(images).reshape(images_shape[0], np.prod(images_shape[1:]))

    # Compute the product of L transpose and L; then get it inverse 
    L_t_L_inv = np.linalg.inv(np.dot(lights.T, lights))

    # Compute the product of L transpose and I 
    L_t_I = np.dot(lights.T, I)

    # Compute G
    G = np.dot(L_t_L_inv, L_t_I)

    # Albedo
    # Reshape the image back from an array of 1980x1080xc to an array of 3 dimensions
    # Get the norm
    G_shape = [G.shape[0]]
    G_shape.extend(images_shape[1:])
    G_albedo = G.reshape(G_shape)
    albedo = np.linalg.norm(G_albedo, axis=0)

    # Normals
    # Use the norm after getting an average on all the colors and divide the norm
    # by G
    G_shape = []
    G_shape.extend(images_shape[1:])
    G_shape.append(3)
    G_normal = np.mean(G.T.reshape(G_shape), axis=2)
    albedo_normal = np.linalg.norm(G_normal, axis=2)
    threshold = 1e-7
    G_normal = G_normal/(1.0*np.maximum(threshold, albedo_normal[:,:,None]))

    # There may be NaN values due to a division by zero
    normals = np.nan_to_num(G_normal)
    return albedo, normals


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    # Create projection matrix
    projection_Matrix = np.dot(K, Rt)

    # Switch from 3D coordinates to homogeneous coordinates
    extra_ones = np.tile([1], points.shape[0]*points.shape[1])
    extra_ones = extra_ones.reshape((points.shape[0], points.shape[1],1))
    h_points = np.concatenate((points, extra_ones), axis=2)
    xs  = np.dot(h_points, projection_Matrix.T)
    # Normalize to convert back from homogeneous coordinates to 2D coordinates
    deno = xs[:,:,2][:,:,np.newaxis]
    normalized_xs = xs / deno

    return normalized_xs[:,:,:2]

def get_pixel_values(img, i, j, u, v, isGrayscale):
    """
    Helper function to return the pixel values based on the position of the window
    """
    if i+u < 0 or i+u >= img.shape[0] or j+v < 0 or j+v >= img.shape[1]:
        return 0 if isGrayscale else np.zeros(img.shape[2])
    return img[i+u][j+v]


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    num_channels = image.shape[-1]
    normalized = np.zeros((image.shape[0], image.shape[1], num_channels * ncc_size**2), dtype = np.float32)
    mid_kernel = ncc_size//2
    height, width, channel = image.shape

    for i in range(mid_kernel, height - mid_kernel):
        for j in range(mid_kernel, width - mid_kernel):
            windows = []
            for u in range(channel):
                window = image[i - mid_kernel : i + mid_kernel + 1, j - mid_kernel : j + mid_kernel + 1, u]
                window = (window - np.mean(window)).flatten()
                windows.append(window.T)
            flatten = np.array(windows).flatten()
            norm = np.linalg.norm(flatten)
            flatten = flatten / norm if norm > 1e-6 else np.zeros(flatten.shape)
            normalized[i,j] = flatten
    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    ncc = np.sum(np.multiply(image1, image2), axis = 2)
    return ncc
