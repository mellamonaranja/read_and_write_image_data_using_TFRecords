# Reading and writing image data using TFRecords
This is an end-to-end example of how to read and write image data using TFRecords. Using an image as input data, you will write the data as a TFRecord file, then read the file back and display the image.

#### step 1. Download images.
```python
kiki  = tf.keras.utils.get_file(
    'Mural_Studio_Ghibli_en_Aguascalientes_con_grafiti_encima_02.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/f/f2/Mural_Studio_Ghibli_en_Aguascalientes_con_grafiti_encima_02.jpg')
```

#### step 2. Read images.
```python
image_string=open(kiki, 'rb').read() #byte type, TensorShape([3064, 4592, 3])
label=image_labels[kiki]
```

#### step 3. Store the raw image string feature as well as the height, width, depth.
```python
def image_example(image_string, label):
  image_shape=tf.io.decode_jpeg(image_string).shape

  feature={
      'height':_int64_feature(image_shape[0]),
      'width':_int64_feature(image_shape[1]),
      'depth':_int64_feature(image_shape[2]),
      'label':_int64_feature(label),
      'image_raw':_bytes_feature(image_string)
  }
#all of the features are now stored in the tf.train.Example message
  return tf.train.Example(features=tf.train.Features(feature=feature))
```

#### step 4. Write the raw image files to tfrecords file.
```python
with tf.io.TFRecordWriter(record_file) as writer:
  for filename, label in image_labels.items():
    image_string=open(filename, 'rb').read()
    tf_example=image_example(image_string, label)
    writer.write(tf_example.SerializeToString())
```

#### step 5. Read the TFRecord file.
```python
image_feature_description={
    'height':tf.io.FixedLenFeature([], tf.int64),
    'width':tf.io.FixedLenFeature([], tf.int64),
    'depth':tf.io.FixedLenFeature([], tf.int64),
    'label':tf.io.FixedLenFeature([], tf.int64),
    'image_raw':tf.io.FixedLenFeature([], tf.string)
}
```

#### step 6. Recover the images from the TFRecord file.
```python
for image_features in parsed_image_dataset:
  image_raw=image_features['image_raw'].numpy()
  display.display(display.Image(data=image_raw))
```

