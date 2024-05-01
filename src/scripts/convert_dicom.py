import argparse

import pydicom as dicom
import os
import cv2
# make it True if you want in PNG format
PNG = False
# Specify the .dcm folder path


def main(args):
    images_path = os.listdir(args.input_folder)

    for n, image in enumerate(images_path):

        if "dicom" in image:
            try:
                ds = dicom.dcmread(os.path.join(args.input_folder, image))

                pixel_array_numpy = ds.pixel_array

                image = image.replace('.dicom', '.png')

                cv2.imwrite(os.path.join(args.input_folder, image), pixel_array_numpy)
            except ValueError as e:
                print(e)
                print(image)
            except AttributeError as e:
                print(e)
                print(image)

            if n % 50 == 0:
                print('{} image converted'.format(n))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder")

    args = parser.parse_args()

    main(args)