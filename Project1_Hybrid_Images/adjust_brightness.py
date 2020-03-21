import cv2
import numpy as np
import argparse

## example usage of this script
## open your system terminal, and type,
##     python adjust_brightness.py --input {path_to_input_image} --output {path_to_output_image} --scale {scale_factor}

## note: set {scale_factor} to 1.0 could help us convert different image formats


def AdjustBrightness(input_img_path, output_img_path, scale_factor):
    img = cv2.imread(input_img_path).astype(np.float64)
    img *= scale_factor
    # truncate values that lie out of [0, 255]
    img[img > 255] = 255

    cv2.imwrite(output_img_path, img.astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adjust Image Brightness (Or Convert Image Format)')
    parser.add_argument('--input', metavar='input_img_path', help='path to input image')
    parser.add_argument('--output', metavar='output_img_path', help='path to output image')
    parser.add_argument('--scale', metavar='scale_factor', help='scale factor applied to image pixels')

    args = parser.parse_args()
    AdjustBrightness(args.input, args.output, float(args.scale))

