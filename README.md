# Deep-Sea-Object-Identifier

##Project Title: 

	Deep Sea Object Identifier 

##Summary:

	This AI model predicts if sonar signals are being bounced against a metal cylinder or a rock. Sixty attributes collected from each bounce determine the type of object being detected. Being able to determine the type of object helps marine vessels avoid collision with dangerous objects. Each attribute is saved as a floating point data. Label for the type of object is saved as ‘M’ (metal) ‘R’ (rock). Although this is a common problem that many other people have provided the solution for, in my model I use classes and methods from TensorFlow instead of writing train and fit models from scratch. I also use TensorFlow optimizer and loss functions instead of creating one myself. Use of classes and methods from TensorFlow make this model easy to follow and modify if needed. 

##Background:
	As mentioned before this model solves the problem of protecting marine vessels from colliding with dangerous objects such as sea mines and explosives. It can be used for both defense and commercial marine activities. It will help safeguard trade and improve economic benefit in general. With ever increasing global trade between countries where most of the merchandise is moved via marine routes use of this model can become very frequent. I believe a more sophisticated model developed on same principles will be used very commonly by commercial vessels. Maritime transport is the backbone of international trade and the global economy. Over 80% of the volume of international trade in goods is carried by sea, and the percentage is even higher for most developing countries. This AI model ensures safe passage of maritime transport. 

##Usage:
	Currently the model just contains a main Python method and few functions that this main method calls. It is a simple model right now and can only be used by calling the main method inside any Python environment like Colab developed by Google. In order to make it more user friendly so it can be used by any novice person it should have an interface that can run on smart phones and also inside an internet browser. Once an interface is developed this could become very useful for companies that develop and monitor marine routes for commercial vessels. It could also be used by the crew of the cargo marine vessels. 

##Project Code:

 Import necessary calsses from tensorflow, panda datasets, models and layers
import tensorflow as tflow
import pandas as pan
import numpy as npy
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#define the one hot encode function to transform the text labels 'R' and 'M' to an array of 1 and zeros.
def hot_encode_one(lbls):
    n_lbls = len(lbls)
    n_unique_labels = len(npy.unique(lbls))
    hot_encode_one = npy.zeros((n_lbls,n_unique_labels))
    hot_encode_one[npy.arange(n_lbls), lbls] = 1
    return hot_encode_one

def main():
  #Import sonar.csv data file into a panda dataset and print the header to verify file was loaded into the panda dataset succesfully
  datafile = pan.read_csv("sonar.csv")
  datafile.head()  

  #separate attributes from lables in their respective arrays
  attributes = datafile.iloc[:,0:60].values
  labels = datafile.iloc[:,60].values

  print(attributes[0:5])
  print(labels[0:5])  

  print(attributes.shape)
  print(labels.shape)

  #Convert text based labels 'R' and 'M' to 1 and zero array
  from sklearn.preprocessing import LabelEncoder
  encoder =  LabelEncoder()
  labels1 = encoder.fit_transform(labels)

  print(labels1)

  #Call hot_encode_one function to change 1 and zero to one hot code value
  one_hot_labels = hot_encode_one(labels1)
  print(one_hot_labels[0:5])
  #print (one_hot_labels)

  #Split the attribute file and its respective labels into training data and test data. 20% of total data will be used as test data to test the accuracy of the
  #model.
  train_attr, test_attr, train_lbls, test_lbls = train_test_split(attributes, one_hot_labels, test_size=0.2, random_state=0)

  #Define model use two layers with 128 and 56 nodes each. Final layer will use softmax.
  model = tflow.keras.Sequential([
          tflow.keras.layers.Dense(128, activation='relu'),
          tflow.keras.layers.Dense(56, activation='relu'),
          tflow.keras.layers.Dense(2, activation='softmax')
          ])
  
  #Complie model use 'rmsprop' optimizer, this optimizer can aslo be chnaged to 'Adam' to see if that helps improve accuracy.
  #Use 'Categorical_crossentropy' to calculate and manage loss since we are using one hot code instead of integer as the label.
  model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  
  #Now train model on the training data
  #We can adjust epoch size here to see if our model trains any better.
  model.fit(train_attr, train_lbls, batch_size=50, epochs=100)

  #evaluate model against test data to calculate loss and accuracy
  loss, accuracy = model.evaluate(test_attr, test_lbls, verbose=0)
  print('Test loss:', loss)
  print('Test accuracy:', accuracy)

  #Make predictions against test data.
  pred_lbls = model.predict(test_attr)
  pred_lbls

  actual = npy.argmax(test_lbls,axis=1)
  predicted = npy.argmax(pred_lbls,axis=1)
  #print(f"Actual: {actual}")
  #print(f"Predicted: {predicted}")
  print (actual)
  print (predicted)

main()  


##Data sources and AI methods:

The project uses a csv file that contains 208 records that are obtained from bouncing a signal against metal cylinder and a rock from different angles and under different conditions. The file contains 60 features and a label that describes the object as M for metal and R if it is a rock. Data is collected by Ashis Bakshi and was taken from his blog. Since we only have 208 records available the accuracy predicted by the model is only at around 80-82%. If we can acquire more data through other resources it will certainly improve the performance of this model. 

##Challenges and what Next?
	This project right now only has a python main function. In order for this program to be used by anyone would require an interface to be developed. An interface that can run on a smart phone or inside an internet browser will make the model to be used by anyone with having a expertise in python. 

	Another challenge is collection of good data. This collection of data will require expertise from someone who is also familiar with the SONAR technology. Actual data collection will require equipment and then actual testing in ocean or another large body of water. This collection of data could not only be costly but will also require highly professional people. 

	With the help of professionals to collect good data and develop an interface for anyone to be able to use this model, this model can be used to solve problems that commercial and defense marine vessels face on every day basis. 
