import pathlib
import IPython.display as display
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# We set some parameters for the model
HEIGHT = 224 #image height
WIDTH = 224 #image width
CHANNELS = 3 #image RGB channels
CLASS_NAMES = ['daisy', 'tulips', 'roses', 'dandelion', 'sunflowers']
NCLASSES = len(CLASS_NAMES)
BATCH_SIZE = 32
SHUFFLE_BUFFER = 10 * BATCH_SIZE
AUTOTUNE = tf.data.experimental.AUTOTUNE

VALIDATION_SIZE = 370
VALIDATION_STEPS = VALIDATION_SIZE // BATCH_SIZE

# Define the function that decodes in the images
def decode_image(image, reshape_dim):
    # JPEG is a compressed image format. So we want to 
    # convert this format to a numpy array we can compute with.
    image = tf.image.decode_jpeg(image, channels=CHANNELS)
    # 'decode_jpeg' returns a tensor of type uint8. We need for 
    # the model 32bit floats. Actually we want them to be in 
    # the [0,1] interval.
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Now we can resize to the desired size.
    image = tf.image.resize(image, reshape_dim)
    
    return image

# The train set actually gives only the paths to the training images.
# We want to create a dataset of training images, so we need a 
# function that can handle this for us.
def decode_dataset(data_row):
    record_defaults = ['path', 'class']
    filename, label_string = tf.io.decode_csv(data_row, record_defaults)
    image_bytes = tf.io.read_file(filename=filename)
    label = tf.math.equal(label_string, CLASS_NAMES)
    return image_bytes, label

# Next we construct a function for pre-processing the images.
def read_and_preprocess(image_bytes, label, augment_randomly=False):
    if augment_randomly: 
        image = decode_image(image_bytes, [HEIGHT + 8, WIDTH + 8])
        # TODO: Augment the image.
        # randomize(height), randomimze(width), randomize(channels)
        # retain everything else
        image = tf.image.random_crop(image, size=[HEIGHT, WIDTH, 3])
        image = tf.image.random_flip_up_down(image)
    else:
        image = decode_image(image_bytes, [HEIGHT, WIDTH])
    return image, label

def read_and_preprocess_with_augmentation(image_bytes, label): 
    return read_and_preprocess(image_bytes, label, augment_randomly=True)

# Now we can create the dataset.
def load_dataset(file_of_filenames, batch_size, training=True):
    # We create a TensorFlow Dataset from the list of files.
    # This dataset does not load the data into memory, but instead
    # pulls batches one after another.
    dataset = tf.data.TextLineDataset(filenames=file_of_filenames).\
        map(decode_dataset)
    
    if training:
        # TODO: Use augmentation here. // DONE
        dataset = dataset.map(read_and_preprocess_with_augmentation).\
            shuffle(SHUFFLE_BUFFER).\
            repeat(count=None) # Infinite iterations
    else: 
        # Evaluation or testing
        dataset = dataset.map(read_and_preprocess).\
            repeat(count=1) # One iteration
            
    # The dataset will produce batches of BATCH_SIZE and will
    # automatically prepare an optimized number of batches while the prior one is
    # trained on.
    return dataset.batch(batch_size).prefetch(buffer_size=AUTOTUNE)