

import tensorflow as tf
import numpy as np
import os

#%%

# you need to change this to your data directory
train_dir = '/Users/y/web/uploads/'

def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    Black_Bent = []
    label_Black_Bent = []
    Charlock = []
    label_Charlock = []
    Cleavers = []
    label_Cleavers = []
    Chickweed = []
    label_Chickweed = []
    Commonwheat = []
    label_Commonwheat = []
    Fathen = []
    label_Fathen = []
    Maize = []
    label_Maize = []
    Scentless = []
    label_Scentless = []
    Shepherds = []
    label_Shepherds = []
    Cranesbill = []
    label_Cranesbill = []
    Suger = []
    label_Suger = []

    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='Black_Bent':
            Black_Bent.append(file_dir + file)
            label_Black_Bent.append(0)
        if name[0]=='Charlock':
            Charlock.append(file_dir + file)
            label_Charlock.append(1)
        if name[0]=='Cleavers':
            Cleavers.append(file_dir + file)
            label_Cleavers.append(2)
        if name[0]=='Chickweed':
            Chickweed.append(file_dir + file)
            label_Chickweed.append(3)
        if name[0]=='Commonwheat':
            Commonwheat.append(file_dir + file)
            label_Commonwheat.append(4)
        if name[0]=='Fathen':
            Fathen.append(file_dir + file)
            label_Fathen.append(5)
        if name[0]=='Maize':
            Maize.append(file_dir + file)
            label_Maize.append(6)
        if name[0]=='Scentless':
            Scentless.append(file_dir + file)
            label_Scentless.append(7)
        if name[0]=='Shepherds':
            Shepherds.append(file_dir + file)
            label_Shepherds.append(8)
        if name[0]=='Cranesbill':
            Cranesbill.append(file_dir + file)
            label_Cranesbill.append(9)
        if name[0]=='Suger':
            Suger.append(file_dir + file)
            label_Suger.append(10)
    print('There are %d Black_Bent\n %d Charlock\n %d Cleavers\n %d Chickweed\n %d Commonwheat\n %d Fathen\n'
          ' %d Maize\n %d Scentless''\n %d Shepherds\n %d Cranesbill\n %d Suger' %
          (len(Black_Bent), len(Charlock), len(Cleavers), len(Chickweed), len(Commonwheat),len(Fathen),len(Maize),
           len(Scentless),len(Shepherds),len(Cranesbill),len(Suger)))

    image_list = np.hstack((Black_Bent,Charlock,Cleavers,Chickweed,Commonwheat,Fathen,Maize,Scentless,Shepherds,Cranesbill,Suger))
    label_list = np.hstack((label_Black_Bent, label_Charlock,label_Cleavers,label_Chickweed,label_Commonwheat,label_Fathen,
                           label_Maize,label_Scentless,label_Shepherds,label_Cranesbill,label_Suger))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    #label_list = list(temp[:, 1])

    # label_list = [int(i) for i in label_list]


    return image_list


#%%

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity = capacity)

    #you can also use shuffle_batch
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch



#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes


#
#
# import matplotlib.pyplot as plt
#
# BATCH_SIZE = 10
# CAPACITY = 1024
# IMG_W = 208
# IMG_H = 208
#
# train_dir = '/Users/y/PycharmProjects/tensorflow/train/train/'
#
# image_list, label_list = get_files(train_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#
#    try:
#        while not coord.should_stop() and i<1:
#
#            img, label = sess.run([image_batch, label_batch])
#
#            # just test one batch
#            for j in np.arange(BATCH_SIZE):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)


#%%
