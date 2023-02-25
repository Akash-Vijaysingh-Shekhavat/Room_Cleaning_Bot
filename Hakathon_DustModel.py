#!/usr/bin/env python
# coding: utf-8

# In[1]:

import RPi.GPIO as GPIO
from gpiozero import Robot
from gpiozero import Motor
from time import sleep
import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy


# In[2]:


# print(os.listdir(r"C:\Users\Madhuri-Amit\Desktop\FloorDataSet"))
print(os.listdir(r"/home/pi/Desktop/DataSet2"))


# In[3]:


train_images = []
train_labels = [] 


# In[4]:


# for directory_path in glob.glob(r"C:\Users\Madhuri-Amit\Desktop/FloorDataSet/*"):
for directory_path in glob.glob(r"/home/pi/Desktop/DataSet2/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path,0) #Reading color images
        img = cv2.resize(img, (200, 200)) #Resize images
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        train_images.append(img)
        train_labels.append(label)


# In[5]:


train_images = np.array(train_images)
train_labels = np.array(train_labels)


# In[6]:


print(train_images)
print(train_labels)


# In[7]:

test_images = []
test_labels = []


# In[8]:


for directory_path in glob.glob(r"/home/pi/Desktop/DataSet2/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path,0)
        img = cv2.resize(img, (200, 200))
        test_images.append(img)
        test_labels.append(fruit_label)


# In[9]:


test_images = np.array(test_images)
test_labels = np.array(test_labels)


# In[10]:


print(test_images)
print(test_labels)


# In[11]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)


# In[ ]:





# In[12]:


print(test_labels_encoded)
print(train_labels_encoded)


# In[13]:


x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded


# In[14]:


import skimage

def feature_extractor(dataset):
   image_dataset = pd.DataFrame()
   for image in range(dataset.shape[0]):  #iterate through each file 
       #print(image)
       #break
       df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
       #Reset dataframe to blank after each loop.
       
       img = dataset[image,:,:]
   ################################################################
   #START ADDING DATA TO THE DATAFRAME
 
               
        #Full image
       #GLCM = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
       GLCM = skimage.feature.graycomatrix(img, [1], [0])       
       GLCM_Energy = skimage.feature.graycoprops(GLCM, 'energy')[0]
       df['Energy'] = GLCM_Energy
       GLCM_corr = skimage.feature.graycoprops(GLCM, 'correlation')[0]
       df['Corr'] = GLCM_corr       
       GLCM_diss = skimage.feature.graycoprops(GLCM, 'dissimilarity')[0]
       df['Diss_sim'] = GLCM_diss       
       GLCM_hom = skimage.feature.graycoprops(GLCM, 'homogeneity')[0]
       df['Homogen'] = GLCM_hom       
       GLCM_contr = skimage.feature.graycoprops(GLCM, 'contrast')[0]
       df['Contrast'] = GLCM_contr


       GLCM2 = skimage.feature.graycomatrix(img, [2], [0])       
       GLCM_Energy2 = skimage.feature.graycoprops(GLCM2, 'energy')[0]
       df['Energy2'] = GLCM_Energy2
       GLCM_corr2 = skimage.feature.graycoprops(GLCM2, 'correlation')[0]
       df['Corr2'] = GLCM_corr2       
       GLCM_diss2 = skimage.feature.graycoprops(GLCM2, 'dissimilarity')[0]
       df['Diss_sim2'] = GLCM_diss2       
       GLCM_hom2 = skimage.feature.graycoprops(GLCM2, 'homogeneity')[0]
       df['Homogen2'] = GLCM_hom2       
       GLCM_contr2 = skimage.feature.graycoprops(GLCM2, 'contrast')[0]
       df['Contrast2'] = GLCM_contr2
      
       GLCM3 = skimage.feature.graycomatrix(img, [5], [0])       
       GLCM_Energy3 = skimage.feature.graycoprops(GLCM3, 'energy')[0]
       df['Energy3'] = GLCM_Energy3
       GLCM_corr3 = skimage.feature.graycoprops(GLCM3, 'correlation')[0]
       df['Corr3'] = GLCM_corr3       
       GLCM_diss3 = skimage.feature.graycoprops(GLCM3, 'dissimilarity')[0]
       df['Diss_sim3'] = GLCM_diss3       
       GLCM_hom3 = skimage.feature.graycoprops(GLCM3, 'homogeneity')[0]
       df['Homogen3'] = GLCM_hom3       
       GLCM_contr3 = skimage.feature.graycoprops(GLCM3, 'contrast')[0]
       df['Contrast3'] = GLCM_contr3

       GLCM4 = skimage.feature.graycomatrix(img, [0], [np.pi/4])       
       GLCM_Energy4 = skimage.feature.graycoprops(GLCM4, 'energy')[0]
       df['Energy4'] = GLCM_Energy4
       GLCM_corr4 = skimage.feature.graycoprops(GLCM4, 'correlation')[0]
       df['Corr4'] = GLCM_corr4       
       GLCM_diss4 = skimage.feature.graycoprops(GLCM4, 'dissimilarity')[0]
       df['Diss_sim4'] = GLCM_diss4       
       GLCM_hom4 = skimage.feature.graycoprops(GLCM4, 'homogeneity')[0]
       df['Homogen4'] = GLCM_hom4       
       GLCM_contr4 = skimage.feature.graycoprops(GLCM4, 'contrast')[0]
       df['Contrast4'] = GLCM_contr4
       
       GLCM5 = skimage.feature.graycomatrix(img, [0], [np.pi/2])       
       GLCM_Energy5 = skimage.feature.graycoprops(GLCM5, 'energy')[0]
       df['Energy5'] = GLCM_Energy5
       GLCM_corr5 = skimage.feature.graycoprops(GLCM5, 'correlation')[0]
       df['Corr5'] = GLCM_corr5       
       GLCM_diss5 = skimage.feature.graycoprops(GLCM5, 'dissimilarity')[0]
       df['Diss_sim5'] = GLCM_diss5       
       GLCM_hom5 = skimage.feature.graycoprops(GLCM5, 'homogeneity')[0]
       df['Homogen5'] = GLCM_hom5       
       GLCM_contr5 = skimage.feature.graycoprops(GLCM5, 'contrast')[0]
       df['Contrast5'] = GLCM_contr5
       
       GLCM6= skimage.feature.graycomatrix(img, [5], [0])       
       GLCM_Energy6 = skimage.feature.graycoprops(GLCM6, 'energy')[0]
       df['Energy6'] = GLCM_Energy6
       GLCM_corr6 = skimage.feature.graycoprops(GLCM6, 'correlation')[0]
       df['Corr6'] = GLCM_corr6       
       GLCM_diss6= skimage.feature.graycoprops(GLCM6, 'dissimilarity')[0]
       df['Diss_sim6'] = GLCM_diss6       
       GLCM_hom6 = skimage.feature.graycoprops(GLCM6, 'homogeneity')[0]
       df['Homogen6'] = GLCM_hom6       
       GLCM_contr6 = skimage.feature.graycoprops(GLCM6, 'contrast')[0]
       df['Contrast6'] = GLCM_contr6
       
       GLCM7 = skimage.feature.graycomatrix(img, [0], [3*np.pi/4])       
       GLCM_Energy7 = skimage.feature.graycoprops(GLCM7, 'energy')[0]
       df['Energy7'] = GLCM_Energy7
       GLCM_corr7 = skimage.feature.graycoprops(GLCM7, 'correlation')[0]
       df['Corr7'] = GLCM_corr7       
       GLCM_diss7 = skimage.feature.graycoprops(GLCM7, 'dissimilarity')[0]
       df['Diss_sim7'] = GLCM_diss7      
       GLCM_hom7 = skimage.feature.graycoprops(GLCM7, 'homogeneity')[0]
       df['Homogen7'] = GLCM_hom7       
       GLCM_contr7 = skimage.feature.graycoprops(GLCM7, 'contrast')[0]
       df['Contrast7'] = GLCM_contr7
       
       GLCM8 = skimage.feature.graycomatrix(img, [3], [np.pi/2])       
       GLCM_Energy8 = skimage.feature.graycoprops(GLCM8, 'energy')[0]
       df['Energy8'] = GLCM_Energy8
       GLCM_corr8 = skimage.feature.graycoprops(GLCM8, 'correlation')[0]
       df['Corr8'] = GLCM_corr8       
       GLCM_diss8 = skimage.feature.graycoprops(GLCM8, 'dissimilarity')[0]
       df['Diss_sim8'] = GLCM_diss8      
       GLCM_hom8 = skimage.feature.graycoprops(GLCM8, 'homogeneity')[0]
       df['Homogen8'] = GLCM_hom8       
       GLCM_contr8 = skimage.feature.graycoprops(GLCM8, 'contrast')[0]
       df['Contrast8'] = GLCM_contr8
       
       #Add more filters as needed
       entropy = shannon_entropy(img)
       df['Entropy'] = entropy
       print(GLCM_Energy,GLCM_corr,GLCM_diss,GLCM_hom,GLCM_contr,entropy)

       
       #Append features from current image to the dataset
       image_dataset = image_dataset.append(df)
       
    
       
   return image_dataset


# In[15]:


image_features = feature_extractor(x_train)
X_for_ML =image_features


# In[17]:


import lightgbm as lgb
 #Class names for LGBM start at 0 so reassigning labels from 1,2,3,4 to 0,1,2,3,4
d_train = lgb.Dataset(X_for_ML, label=y_train)


# In[ ]:





# In[18]:


lgbm_params = {'learning_rate':0.05, 'boosting_type':'dart',    
              'objective':'multiclass',
              'metric': 'multi_logloss',
              'num_leaves':100,
              'max_depth':15,
              'num_class':2}  #no.of unique values in the target class not inclusive of the end value
print(lgbm_params)


# In[19]:


lgb_model = lgb.train(lgbm_params, d_train, 100) #50 iterations. Increase iterations for small learning rates


# In[ ]:





# In[20]:


test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

#Predict on test
test_prediction = lgb_model.predict(test_for_RF)
test_prediction=np.argmax(test_prediction, axis=1)
#Inverse le transform to get original label back. 
test_prediction = le.inverse_transform(test_prediction)


# In[21]:


from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))


# In[22]:


print("Precision Score: ",metrics.precision_score(test_labels, test_prediction,average='macro'))
print("Recall Score: ",metrics.recall_score(test_labels, test_prediction,average='weighted'))
print("F1 Score: ",metrics.f1_score(test_labels, test_prediction,average='micro'))


# In[23]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, test_prediction)

fig, ax = plt.subplots(figsize=(6,6))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)


# In[25]:

robot = Robot(left=(26,20),right=(19,16))
mopper =Motor(22,23,27)
GPIO.setmode(GPIO.BCM)
GPIO.setup(2,GPIO.OUT)

import cv2
cap = cv2.VideoCapture(0)
     
while(True):
    #Capture frame-by-frame
    ret, img = cap.read()

     #Pass frame through feature extractor, reshape and predict
    input_img = np.expand_dims(img, axis=0)
    input_img = np.reshape(input_img, (input_img.shape[0], input_img.shape[1], -1))

    input_img_features=feature_extractor(input_img)
    input_img_features = np.expand_dims(input_img_features, axis=0)
    input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
    img_prediction = lgb_model.predict(input_img_for_RF)
    img_prediction=np.argmax(img_prediction, axis=1)
    img_prediction = le.inverse_transform([img_prediction])
    img_prediction = str(img_prediction)
    print(img_prediction)
    if img_prediction == "['/home/pi/Desktop/DataSet2/Dust_Present']":
        img_prediction = 'Dust_Present'
        robot.forward()
        mopper.forward()
        GPIO.output(2,GPIO.HIGH)
        print(2)
    elif img_prediction == "['/home/pi/Desktop/DataSet2/Dust_Absent']":
        img_prediction = 'Dust_Absent'
        print(3)
        robot.stop()
        mopper.stop()
        GPIO.output(2,GPIO.LOW)

    # Display the resulting frame
    print("img_prediction", img_prediction)
    print("\n")
    cv2.putText(img, "Prediction: "+ img_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('WebCam', img)
        
#     
#     img_prediction = lgb_model.predict(input_img_for_RF)
#     img_prediction=np.argmax(img_prediction, axis=1)
#     img_prediction = le.inverse_transform([img_prediction])
#     
#     print(img_prediction)
#     print("Dust Appp")
#     
    
    def delay():
        print("Delay")
        time.sleep(100)
    def move():
        robot.forward()
        print("Move")
    def vaccum():
        print("Vaccum on")
    def mopp():
        print("Mopping on")
        mopper.forward()
    def stop():
        print("Stop all")
        robot.stop()
        
    def wheel():
        robot.forward()
        mopper.forward()
        sleep(10)
        robot.stop()
        mopper.stop()
    
    if img_prediction == "Dust_Absent":
        print("Absent")
        stop()
        
    else:
        print("Present")
        
        
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
        
        
# Release the webcam and close windows


# In[ ]:




