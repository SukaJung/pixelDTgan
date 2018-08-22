import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
class Dataset(object):
    def __init__(self):
        self.size = 64
        self.channel = 3
        self.dataset = []
        self.read_dataset()


    def read_dataset(self):
        data_path = 'train.tfrecord'  # address to save the hdf5 file
        feature = {'train/img': tf.FixedLenFeature([], tf.string),
                       'train/ass_label': tf.FixedLenFeature([], tf.string)}

        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example, features=feature)

        # self.img = tf.decode_raw(features['train/img'], tf.float32)
        # self.ass_label = tf.decode_raw(features['train/ass_label'], tf.float32)
        # self.noass_label = tf.decode_raw(features["train/noass_label"],tf.float32)

        # self.img = tf.reshape(self.img, [self.size, self.size, self.channel])
        # self.ass_label = tf.reshape(self.ass_label, [self.size, self.size, self.channel])
        # self.noass_label = tf.reshape(self.noass_label,[self.size, self.size, self.channel])

        self.reconstructed_images = []

        record_iterator = tf.python_io.tf_record_iterator(path=data_path)
        count = 0
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            img_string = (example.features.feature['train/img']
                          .bytes_list
                          .value[0])
            ass_label_string = (example.features.feature['train/ass_label']
                          .bytes_list
                          .value[0])
            noass_label_string = (example.features.feature['train/noass_label']
                          .bytes_list
                          .value[0])

            img_1d = np.fromstring(img_string, dtype=np.float64)
            ass_label_1d = np.fromstring(ass_label_string, dtype=np.float64)
            noass_label_1d = np.fromstring(noass_label_string, dtype=np.float64)

            print(count)
            count+=1
            reconstructed_img = img_1d.reshape(64, 64, 3)
            reconstructed_ass_label = ass_label_1d.reshape(64, 64, 3)
            reconstructed_noass_label = noass_label_1d.reshape(64, 64, 3)
            
            self.reconstructed_images.append([reconstructed_ass_label,reconstructed_noass_label,reconstructed_img])
        print(len(self.reconstructed_images))
        print("Complete read dataset")

    def get_batch(self,batch_size= 128):
        img, ass_label,noass_label = tf.train.shuffle_batch([self.img, self.ass_label,self.noass_label], batch_size=batch_size, capacity=228, num_threads=1,
                                                    min_after_dequeue=10)
        return ass_label,noass_label,img


    def getdata(self,batch_size=128):
        self.dataset= np.array(self.reconstructed_images)
        random.shuffle(self.dataset)
        print(self.dataset.shape)
        return self.dataset[:batch_size,0],self.dataset[:batch_size,1],self.dataset[:batch_size,2]