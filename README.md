# Bank-Recognition-Model

## Description 
------------------------------------
Objective : To classify the banks based on the pictures of banks (different angles, lighting, timing, etc) that we collected as a team 
1. Datasets collected by ourselves - CIMB Bank, MBSB Bank, Citibank, Agrobank, Maybank.
2. Models : SVM, CNN

## Sample Data
----------------
Pre-processing steps :
1. Crop to a ratio of 1:1
2. Resize 128 x 98 pixels
3. Convert to grayscale

Before Pre-processing : 

![image](https://github.com/eethiing/Bank-Recognition-Model/assets/85276977/c17f4e99-8df8-4975-9058-ffb917d16dbb)
![image](https://github.com/eethiing/Bank-Recognition-Model/assets/85276977/b8b0f6f6-8626-43b2-bb26-77100cadeb2f)
![image](https://github.com/eethiing/Bank-Recognition-Model/assets/85276977/7146c70e-0939-4d63-a434-0348044fb8b0)

After pre-processing :

![image](https://github.com/eethiing/Bank-Recognition-Model/assets/85276977/f4e323d0-4dd6-43a5-aaf1-3cd8a37434ff)
![image](https://github.com/eethiing/Bank-Recognition-Model/assets/85276977/1af64560-107b-4a25-84ca-1c8f20d3f180)
![image](https://github.com/eethiing/Bank-Recognition-Model/assets/85276977/1cc66f8a-2e41-49ff-9642-cd0b7ec50cfe)

## Experiment Results 
----------
From the confusion matrix, we can also see that the testing data that we have used is perfectly classified to each of their labels without any mistakes. All of the 4 images from each object are correctly labeled when we are using the CNN model. The classification report also shows the precision, recall and f1 score are all equivalent to 1.0. From the training and validation accuracy graph, we can conclude that as the number of epochs is reaching to 10, the accuracy is rising to almost 1.0 whereas from the training and validation loss graph, we can conclude that as the number of epochs reaching to 10, the loss is decreasing to 0. However, since the dataset is small, CNN is not a suitable model for this. 

![image](https://github.com/eethiing/Bank-Recognition-Model/assets/85276977/c2510d1b-81ba-430c-aa57-6a6338883b80)
![image](https://github.com/eethiing/Bank-Recognition-Model/assets/85276977/95644502-955a-47a7-9402-143a1f554511)

Based on the classification report, the results of SVM is comparable to CNN. In observation with training and cross-validation score graph for SVM, training score was kept at a constant 1.0. However, there was a significant difference in cross validation score. The cross validation score mounted greatly, adjusted itself and showed a slight raise after. Through the end, it ceased at 0.7. 

![image](https://github.com/eethiing/Bank-Recognition-Model/assets/85276977/2396e60c-0dfa-486f-873e-5c1d07a6b372)
![image](https://github.com/eethiing/Bank-Recognition-Model/assets/85276977/54b03ea1-aab9-418c-8096-0b4500fd5590)








