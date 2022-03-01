import streamlit as st
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn import metrics

st.sidebar.title("Pattern Recognition Assignment")

st.sidebar.info("**Project Designed By:** \n\n Chow Zheng Xuan  1191302602 \n\n  Wong Ee Thiing  1181100509 \n\n  Mohamad Hizzudin Bin Mohamad Hanafiah  1181101153  \n\n  Vinod A/L Raghu  1181102022")

st.title('Bank Recognition System')

st.write("Reference <br> 1 - Citibank <br> 2 - MBSB Bank <br> 3 - Agro Bank <br> 4 - CIMB Bank <br> 5 - Maybank", unsafe_allow_html=True)

classifier = st.sidebar.selectbox('Which Model do you want to use?',('SVM' , 'CNN'))

def trainSVM():
    t=time.time()
    
    train_images = []
    train_label = []
    FOLDER_NAME = []
    DATADIR = "Bank/Training/"
    
    for i in range(1,6) :
        FOLDER_NAME.append(str(i))
    
    def load_train_images(folder,i):
        for filename in os.listdir(folder):
             if any([filename.endswith(x) for x in ['.jpeg', '.jpg']]):
                 img = cv2.imread(os.path.join(folder, filename))
             train_images.append(img)
             train_label.append(i)
        return train_images, train_label
     
    for i in range(1,6):
          folders = "Bank/Training/" + str(i) #change folder name
          training_images= load_train_images(folders,i)
          
    data_gray = [cv2.cvtColor(train_images[i] , cv2.COLOR_BGR2GRAY) for i in range(len(train_images))]
    
    Labels = np.array(train_label).reshape(len(train_label),1)
    
    gray_features = np.array(data_gray)
    
    pixels = gray_features.flatten().reshape(80, 12544)
    g_df = np.hstack((pixels,Labels))
    
    Labels_train = g_df[:,-1]
    
    pca = PCA(n_components=npca , svd_solver='randomized' , whiten=True).fit(pixels)
     
    pixels_pca = pca.transform(pixels)
     
    svm = SVC(kernel='rbf' , class_weight='balanced' , C=c , gamma=0.01)
    svm.fit(pixels_pca , Labels_train)
     
    print("Training time: %0.3fs" % (time.time() - t))
    
    #Testing Phases 
    t2=time.time()
     
    test_images2 = []
    test_label2 = []
    FOLDER_NAME2 = []
    DATADIR2 = "Bank/Testing/"
     
    for i in range(1,6) :
         FOLDER_NAME2.append(str(i))
    
    def load_test_images(folder2,i):
        for filename in os.listdir(folder2):
            if any([filename.endswith(x) for x in ['.jpeg', '.jpg']]):
                img = cv2.imread(os.path.join(folder2, filename))
            test_images2.append(img)
            test_label2.append(i)
        return test_images2, test_label2
    
    for i in range(1,6):
        folders2 = "Bank/Testing/" + str(i) #change folder name
        test_images= load_test_images(folders2,i)
    
    test_data_gray = [cv2.cvtColor(test_images2[i] , cv2.COLOR_BGR2GRAY) for i in range(len(test_images2))]
    
    Labels_test = np.array(test_label2).reshape(len(test_label2),1)
    
    test_gray_features = np.array(test_data_gray)
    
    test_pixels = test_gray_features.flatten().reshape(20, 12544)
    test_g_df = np.hstack((test_pixels,Labels_test))
    
    test_pixels_pca = pca.transform(test_pixels)
    
    train_predictions = svm.predict(pixels_pca)
    test_predictions = svm.predict(test_pixels_pca)
    train_score = (svm.score(pixels_pca , Labels_train))
    test_score = (svm.score(test_pixels_pca , Labels_test))
    
    cm = confusion_matrix(Labels_test , test_predictions)
    cr = classification_report(Labels_test , test_predictions)
    
    print("Testing time: %0.3fs" % (time.time() - t2))
    
    st.subheader('Train Score: {}'.format(train_score))
    st.subheader('Test Score: {}'.format(test_score))
    st.subheader('Confusion matrix: ')
    st.write(cm)
    st.subheader('Classification Report: ')
    st.write(cr)
    print(cr)
    
    def plot_learning_curve(svm, title, pixels_pca , Labels_train, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(svm, pixels_pca , Labels_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
        
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
         
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
        
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
        
        plt.legend(loc="best")
        
        return plt
    
    title = "Learning Curves"
    cv = ShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    #estimator = SVC(gamma=0.001)
    plot_learning_curve(svm, title, pixels_pca , Labels_train, (0.0, 1.01), cv=cv, n_jobs=4)
    
    #plt.show()  
    plt.savefig("LearningCurve.png")
    st.image("LearningCurve.png")
    
    #Visualise Some Misclassified Pic
    index = 0
    misclassifiedIndexes = []
    for label, predict in zip(Labels_test , test_predictions):
        if label != predict:
            misclassifiedIndexes.append(index)
        index = index+1
        
    plt.figure(figsize=(32, 24))
    for plotIndex, wrongIndex in enumerate(misclassifiedIndexes[0:9]):
        
        plt.subplot(3,3, plotIndex+1)
        plt.imshow(np.reshape(test_pixels[wrongIndex], (98,128)))
        plt.title('Predicted: {}, Actual: {}'. format(test_predictions[wrongIndex], Labels_test[wrongIndex]), fontsize=24)
        
        
    plt.savefig("Wrongindex.png")
    st.subheader('Wrong Prediction: ')
    st.image("Wrongindex.png")





def trainCNN():
    t=time.time()
    
    #Import Training Image
    train_images = []
    train_label = []
    FOLDER_NAME = []
    DATADIR = "Bank/Training/"
    
    for i in range(1,6) :
        FOLDER_NAME.append(str(i))
        
    def load_train_images(folder,i):
        for filename in os.listdir(folder):
            if any([filename.endswith(x) for x in ['.jpeg', '.jpg']]):
                img = cv2.imread(os.path.join(folder, filename))
            train_images.append(img)
            train_label.append(i)
        return train_images, train_label
    
    for i in range(1,6):
        folders = "Bank/Training/" + str(i) #change folder name
        training_images= load_train_images(folders,i)
        
    #Import Testing Image
    test_images2 = []
    test_label2 = []
    FOLDER_NAME2 = []
    DATADIR2 = "Bank/Testing/"
    
    for i in range(1,6) :
        FOLDER_NAME2.append(str(i))
        
    def load_test_images(folder2,i):
        for filename in os.listdir(folder2):
            if any([filename.endswith(x) for x in ['.jpeg', '.jpg']]):
                img = cv2.imread(os.path.join(folder2, filename))
            test_images2.append(img)
            test_label2.append(i)
        return test_images2, test_label2
    
    for i in range(1,6):
        folders2 = "Bank/Testing/" + str(i) #change folder name
        test_images= load_test_images(folders2,i)
        
    #Train & Test
    X_train = np.array(train_images)
    Y_train = np.array(train_label)
    X_test = np.array(test_images2)
    Y_test = np.array(test_label2)
    
    model = tf.keras.Sequential()

    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',input_shape=(98,128,3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())  #Convert our 3D feature maps into 1D feature vectors

    model.add(tf.keras.layers.Dense(256,activation= 'relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(128,activation= 'relu'))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(161,activation='softmax'))
    
    model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    
    hist=model.fit(X_train,Y_train,epochs=10,validation_data=(X_test, Y_test))
    
    train_score = hist.history['accuracy']
    test_score = hist.history['val_accuracy']
    
    #train_predictions = model.predict(X_train)
    test_predictions_ = model.predict(X_test)
    test_predictions = np.argmax(test_predictions_, axis=1)
    
    print("Training time: %0.3fs" % (time.time() - t))
    
    cm = confusion_matrix(Y_test , test_predictions)
    cr = classification_report(Y_test , test_predictions)
    
    st.subheader('Train Score: {}'.format(train_score[9]))
    st.subheader('Test Score: {}'.format(test_score[9]))
    st.subheader('Confusion matrix: ')
    st.write(cm)
    st.subheader('Classification Report: ')
    st.write(cr)
    print(cr)
    
    acc=hist.history['accuracy']
    val_acc=hist.history['val_accuracy']

    loss=hist.history['loss']
    val_loss=hist.history['val_loss']

    epochs_range=range(10)
    plt.figure(figsize=(8,8))
    
    plt.subplot(1,2,1)
    plt.plot(epochs_range,acc,label='Training Accuracy')
    plt.plot(epochs_range,val_acc,label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1,2,2)
    plt.plot(epochs_range,loss,label='Training Loss')
    plt.plot(epochs_range,val_loss,label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig("CNN_Graph.png")
    st.image("CNN_Graph.png")
    
    #Visualise Some Misclassified Pic
    index = 0
    misclassifiedIndexes = []
    for label, predict in zip(Y_test , test_predictions):
        if label != predict:
            misclassifiedIndexes.append(index)
        index = index+1
        
    plt.figure(figsize=(32, 24))
    for plotIndex, wrongIndex in enumerate(misclassifiedIndexes[0:9]):
        
        plt.subplot(3,3, plotIndex+1)
        plt.imshow(train_images[wrongIndex])
        plt.title('Predicted: {}, Actual: {}'. format(test_predictions[wrongIndex], Y_test[wrongIndex]), fontsize=24)
        
        
    plt.savefig("Wrongindex.png")
    st.subheader('Wrong Prediction: ')
    st.image("Wrongindex.png")





if classifier == 'SVM':
    st.sidebar.title("Support Vector Machine")
    npca = st.sidebar.slider(label='Chose number of component of PCA' , min_value=1, max_value=80)
    c = st.sidebar.slider(label='Chose value of C' , min_value=1, max_value=1000)
    
    if st.sidebar.button('Train SVM'): 
     trainSVM()
    
elif classifier == 'CNN':
    st.sidebar.title("Convolutional Neural Network")
    
    if st.sidebar.button('Train CNN'):
        trainCNN()
