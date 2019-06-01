#!/usr/bin/env python3
"""
mnist_interpreter: Show images and corresponding label from the testing set

Here's an overview of the file format, so you will have a basic understanding
of the source code.

IDX file contains a header to indicate how the matrix is stored, and it is easy
to explain the header structure using C language.

struct idx_magic {
    unsigned char zero[2];
    unsigned char type_no;  // Type number, 0x8 for unsigned char
    unsigned char ndims;    // How many dimensions does the matrix have
};

struct idx_header {
    struct idx_magic magic;
    size_t sz_dim0;     // For MNIST, # of images
    size_t sz_dim1;     // For MNIST, # of rows
    size_t sz_dim2;     // For MNIST, # of columns
    ...
    size_t sz_dimN;
    char data[];
};

NOTE: The numbers are stored using big endian bytecode (which is popular among
non-intel processors)

So the corresponding format for Python's struct module shall be '>xxBB#i', where
`#' stands for the number of dimensions.
"""
import gzip
import struct
import cv2
import sys
import numpy as np

images_fmt=">xxBB3i"    # 3 dimensions in the images file, so 3 `i's
labels_fmt=">xxBBi"     # 1 dimension, so only one `i'
set_types=["train","test"]

if len(sys.argv) < 2:
    print("Usage: %s [train|test]" % sys.argv[0])
    sys.exit(1)

set_type=sys.argv[1]
if set_type not in set_types:   # Make sure user enters proper stuff
    print("%s: invalid set type" % sys.argv[1])
    sys.exit(1)

if set_type=="test":
    set_type="t10k"

try:
    images_file=gzip.open("./%s-images-idx3-ubyte.gz" % set_type)
    raw_images=images_file.read()
    images_file.close()     # Close the file to free the resources
    labels_file=gzip.open("./%s-labels-idx1-ubyte.gz" % set_type)
    raw_labels=labels_file.read()
    labels_file.close()
except FileNotFoundError:
    print("error: MNIST %sing image/label set is missing missing" % sys.argv[1])
    exit(-1)
type_no,ndims,images,rows,cols=struct.unpack_from(images_fmt,raw_images,0)
# Make sure we are reading a valid MNIST IDX file
assert(ndims == 3)
assert(type_no == 8)
print("Found %d images with resolution %dx%d" % (images,cols,rows))
type_no,ndims,labels=struct.unpack_from(labels_fmt,raw_labels,0)
# (Like what was explained above)
assert(ndims == 1)
assert(type_no == 8)
assert(labels == images)

img_size=rows*cols
offset=struct.calcsize(images_fmt)
offset2=struct.calcsize(labels_fmt)
img_fmt=">"+str(img_size)+"B"   # How big is one image block?
# Now we are iterating through each image and label them in the viewer
for i in range(images): # Since `labels' shall be equal to `images'
    img=np.array(struct.unpack_from(img_fmt,raw_images,offset),np.uint8)
    img=img.reshape(rows,cols)
    label=struct.unpack_from("B",raw_labels,offset2)
    # In order to let OpenCV show it, we need some conversions
    cvimg=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # Also enlarge the image, since we are going to put text on it
    cvimg=cv2.resize(cvimg,(28*4,28*4))
    cv2.putText(cvimg,"%d" % label,(5,100),cv2.FONT_HERSHEY_SIMPLEX,
            1,(255,0,0),2)  # OpenCV uses BGR, so (255,0,0) stands for blue
    cv2.imshow("MNIST data (K=Pause, Q=Quit)",cvimg)
    key=cv2.waitKey(50) & 0xFF  # We do not need data of higher bytes
    if key==ord('q'): break
    if key==ord('k'):
        if (cv2.waitKey(0) & 0xFF)==ord('q'): break
    offset+=struct.calcsize(img_fmt)    # Jump to the next image
    offset2+=1  # Increment by one since the size of each label container is 1
