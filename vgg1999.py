from keras.models import load_model
import numpy as np
import keras
import cv2
import os

def accuracy():
    #type: (void)->str
    img_width, img_height = 100, 100

    model = load_model('/Users/y/web/plant_trained_model2.h5')

    opt = keras.optimizers.adam(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,metrics=['accuracy'])

    result_dir = '/Users/y/web/uploads'

    #获得文件夹内所有文件
    lists = os.listdir(result_dir)

    lists.sort(key=lambda fn: os.path.getmtime(result_dir+'/' + fn))
    #把文件路径和文件名链接到一起
    file = os.path.join(result_dir, lists[-1])

    # for root, dirs, files in os.walk('/Users/y/web/uploads/'):
    #     for name in files:
    #         if(name != file):
    #             os.remove(os.path.join(root, name))

    print(file)

    img = cv2.imread(file)
    img1 = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
    images = np.array(img1)
    images = images.astype(np.float32)
    images = np.expand_dims(images, axis=0)

    classes = model.predict(images)
    print (classes)
    i=0
    y = classes.astype(np.int32)
    new = y.tolist()
    print (y)
    item = new[0]
    for new in item:
        if new==0:
            i=i+1
        elif new==1:
            break
    if  i== 0:
        x='The plant seedling you test is black glasss'
        print(x)
        return x
    elif i == 1:
        x = 'The plant seedling you test is charlock'
        print(x)
        return x
    elif i == 2:
        x = 'The plant seedling you test is cleavers'
        print(x)
        return x
    elif i == 2:
        x = 'The plant seedling you test is Common Chickweed'
        print(x)
        return x
    elif i == 3:
        x = 'The plant seedling you test is Cleavers'
        print(x)
        return x
    elif i == 4:
        x = 'The plant seedling you test is Common wheat'
        print(x)
        return x
    elif i == 5:
        x = 'The plant seedling you test is Fat Hen'
        print(x)
        return x
    elif i == 6:
        x = 'The plant seedling you test is Loose Silky-bent'
        print(x)
        return x
    elif i == 7:
        x = 'The plant seedling you test is Maize'
        print(x)
        return x
    elif i == 8:
        x = 'The plant seedling you test is Scentless Mayweed'
        print(x)
        return x
    elif i == 9:
        x = 'The plant seedling you test is Shepherds Purse'
        print(x)
        return x
    elif i == 10:
        x = 'The plant seedling you test is Small-flowered Cranesbill'
        print(x)
        return x
    elif i == 11:
        x = 'The plant seedling you test is Sugar beet'
        print(x)
        return x
#accuracy()

#print(accuracy())
