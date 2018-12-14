# Seedlings
Computer vision project to determine plant type

The idea of the whole project is to classify 12 kinds of seedlings as accurate as we can. However, there is two kinds of plants are very similar: Black_grass and Loose Silky_Bent, and by using the AutoML as a benchmarking, we can also find that it is a little hard to detect them even using AutoMl.
![AutoML_results](AutoML_results.png)
Precision tells us, from all the test examples that were assigned a label, how many actually were supposed to be categorized with that label. Recall tells us, from all the test examples that should have had the label assigned, how many were actually assigned the label.
![confusion_matrix](confusion_matrix.png)
So, after discussion, we decided to use two different models to detect all 12 labels. First we combine the Black_grass and Loose Silky_Bent dataset together as a Black_Bent label, and train all images with a simple model, say a 7_layers one. And use another complex model, for instance, vgg19, to classify Black_grass and Loose Silky_Bent. After the first training, we can have a good accuracy and then seperate the image of Black_Bent out, use the complex model to classify them.

Here is our result of first model (7 layers) with 11 labels.
![training2](training2.png)
![accuracy2](accuracy2.png)
Here is our result of first model (7 layers) with 12 labels.
![training](training.png)
![accuracy](accuracy.png)
As we can see, the 11_label_model's lose is much lower than the 12_label_model's.

Next step, we're gonna to build a vgg19 model to classify the Black_grass and Loose Silky_Bent. 

############################# This is Update line #########################

Actually, we r confused by the result that working with a 7-layer-model we can get such a good result, and then we checked again, fingding that we used images in training set to test, and after we retrained and use some totally new images to test, the accuracy decrease to about 68%, which cannot meet our request. And we used Vgg19 to train and test, the result is good, bigger than 90%. We'll update the model in a few days.

######################### This is Update Line 12.13.2018 #####################

So, finally to use and test our project, u need to download these files: main.py, training.py,input_data.py, model.py, vgg1999.py, and our model file. Because we built the website on our localhost server, perhaps it's hard to fix the enviornment on ur computer, and some file address also needed to change to ur own pc address.

There a two models in our final push. One is based on vgg19, and another one is built by ourselves with seven layers. And we mainly focused on the vgg19 model, which is much more accurate and useful when we test images online. Separate the given images for training and validation with an 80%/20% split, we can achieve an accuracy of 88.33% on the validation set. Using all labeled data for training and making use of little data augmentation, we can obtain an accuracy of 83.06% on the test set. 

The matrixs below represent our accuracy. First one is without normalization, and the second one is after normalizing process.

![image](https://github.com/tomkelly110/Seedlings/blob/master/result/matrix1.png)
![image](https://github.com/tomkelly110/Seedlings/blob/master/result/matrix2.png)

