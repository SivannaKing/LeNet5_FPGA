#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@AUTHOR     WZX
@EMAIL      wuzhong_xing@126.com
@TIME&LOG   2022/4/12 - download - wzx
            -----------------
            basic function

            2022/4/13 - modify - wzx
            -----------------
            modify annotation

 TODO       google style annotation
@FUNC       convert mnist .gz to .json
@USAGE      >>> python convert_mnist_to_json.py
            under dir [LeNet5_Pytorch]
'''


# The MNIST dataset has labels (which denote the correct value of the
# handwritten digit) and pixel values separate. The images are a 28 by 28 pixel
# grid of values that range from 0 (white) to 255 (black) with in-between
# values being shades of gray.

def convert(image_file, label_file, output_file, num_images):
    labels = open(label_file, "rb")
    images = open(image_file, "rb")
    output = open(output_file, "w")

    # Discards header info
    images.read(16)
    labels.read(8)

    # Writes image and label data to output file in JSON format
    output.write('{\n  "imageCount": ' + str(num_images) + ',\n  "images": [\n')
    for i in range(num_images):
        label = ord(labels.read(1))  # ord:return ASCll value
        output.write('    {\n      "digit": ' + str(label) + ',\n')
        output.write('      "pixels": [')
        for j in range(784):
            value = ord(images.read(1))
            if j == 0:
                output.write(str(value))
            else:
                output.write(', ' + str(value))
        output.write(']\n    }')
        if i < num_images - 1:
            output.write(',\n')
    output.write('\n  ]\n}')

    # Closes files when read/write operations are finished
    images.close()
    output.close()
    labels.close()


def main():
    convert(
        # This should match the path of the unzipped MNIST image binary
        "./data/mnist/MNIST/raw/train-images-idx3-ubyte",

        # This should match the path of the unzipped MNIST label binary
        "./data/mnist/MNIST/raw/train-labels-idx1-ubyte",

        # Change this to what you want your JSON object to be called
        "mnistTest.json",

        # Change this value to the number of images you want output to your
        # JSON object
        10
    )

    print("Convert Finished!")


if __name__ == "__main__":
    main()
