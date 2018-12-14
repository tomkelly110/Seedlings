
import os
import numpy as np
import tensorflow as tf
import input_data
import model


#%%

N_CLASSES = 11
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 16
CAPACITY = 512
MAX_STEP = 4400 # with current parameters, it is suggested to use MAX_STEP>4k
learning_rate = 0.0005 # with current parameters, it is suggested to use learning rate<0.0005



from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train):
   '''Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   image = Image.open(img_dir)
   # plt.imshow(image)
   # plt.show()
   image = image.resize([208, 208])
   image = np.array(image)
   return image


def maintain_one_image():
    result_dir = '/Users/y/web/uploads/'

    #获得文件夹内所有文件
    lists = os.listdir(result_dir)

    lists.sort(key=lambda fn: os.path.getmtime(result_dir+'/' + fn))
    #把文件路径和文件名链接到一起
    file = lists[-1]

    for root, dirs, files in os.walk('/Users/y/web/uploads/'):
        for name in files:
            if(name != file):
                os.remove(os.path.join(root, name))

def evaluate_one_image():
   '''Test one image against the saved models and parameters
   '''
   maintain_one_image()
   # you need to change the directories to yours.

   train_dir = '/Users/y/web/uploads/'
   ##
   train = input_data.get_files(train_dir)
   image_array = get_one_image(train)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 11

       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 208, 208, 3])
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[208, 208, 3])

       # you need to change the directories to yours.
       logs_train_dir = '/Users/y/web/train/logs/train/'

       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           x =''
           if max_index==0:
               m = "%.6f " %prediction[:, 0]
               x = 'This is a Black_grass or Loose Silky-bent with possibility ' + m
               return x
           if max_index==1:
               m = "%.6f " %prediction[:, 1]
               x = 'This is a Charlock with possibility ' + m
               return x
           if max_index==2:
               m = "%.6f " %prediction[:, 2]
               x = 'This is a Cleavers with possibility ' + m
               return x
           if max_index==3:
               m = "%.6f " %prediction[:, 3]
               x = 'This is a Chickweed with possibility ' + m
               return x
           if max_index==4:
               m = "%.6f " %prediction[:, 4]
               x = 'This is a Commonwheat with possibility ' + m
               return x
           if max_index==5:
               m = "%.6f " %prediction[:, 5]
               x = 'This is a Fathen with possibility ' + m
               return x
           if max_index==6:
               m = "%.6f " %prediction[:, 6]
               x = 'This is a Maize with possibility ' + m
               return x
           if max_index==7:
               m = "%.6f " %prediction[:, 7]
               x = 'This is a Scentless with possibility ' + m
               return x
           if max_index==8:
               m = "%.6f " %prediction[:, 8]
               x = 'This is a Shepherds with possibility ' + m
               return x
           if max_index==9:
               m = "%.6f " %prediction[:, 9]
               x = 'This is a Cranesbill with possibility ' + m
               return x
           else:
               m = "%.6f " %prediction[:, 10]
               x = 'This is a Suger with possibility ' + m
               return x
#evaluate_one_image()

