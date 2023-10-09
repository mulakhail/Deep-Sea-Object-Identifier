# Deep-Sea-Object-Identifier
## Summary:

	This AI model predicts if sonar signals are being bounced against a metal cylinder or a rock. Sixty attributes collected from each bounce determine the type of object being detected. Being able to determine the type of object helps marine vessels avoid collision with dangerous objects. Each attribute is saved as a floating point data. Label for the type of object is saved as ‘M’ (metal) ‘R’ (rock). Although this is a common problem that many other people have provided the solution for, in my model I use classes and methods from TensorFlow instead of writing train and fit models from scratch. I also use TensorFlow optimizer and loss functions instead of creating one myself. Use of classes and methods from TensorFlow make this model easy to follow and modify if needed. 

 “Building AI course project”

## Background:
	As mentioned before this model solves the problem of protecting marine vessels from colliding with dangerous objects such as sea mines and explosives. It can be used for both defense and commercial marine activities. It will help safeguard trade and improve economic benefit in general. With ever increasing global trade between countries where most of the merchandise is moved via marine routes use of this model can become very frequent. I believe a more sophisticated model developed on same principles will be used very commonly by commercial vessels. Maritime transport is the backbone of international trade and the global economy. Over 80% of the volume of international trade in goods is carried by sea, and the percentage is even higher for most developing countries. This AI model ensures safe passage of maritime transport. 

## Usage:
	Currently the model just contains a main Python method and few functions that this main method calls. It is a simple model right now and can only be used by calling the main method inside any Python environment like Colab developed by Google. In order to make it more user friendly so it can be used by any novice person it should have an interface that can run on smart phones and also inside an internet browser. Once an interface is developed this could become very useful for companies that develop and monitor marine routes for commercial vessels. It could also be used by the crew of the cargo marine vessels. 

## Data sources and AI methods:

The project uses a csv file that contains 208 records that are obtained from bouncing a signal against metal cylinder and a rock from different angles and under different conditions. The file contains 60 features and a label that describes the object as M for metal and R if it is a rock. Data is collected by Ashis Bakshi and was taken from his blog. Since we only have 208 records available the accuracy predicted by the model is only at around 80-82%. If we can acquire more data through other resources it will certainly improve the performance of this model. 

## Challenges and what Next?
	This project right now only has a python main function. In order for this program to be used by anyone would require an interface to be developed. An interface that can run on a smart phone or inside an internet browser will make the model to be used by anyone with having a expertise in python. 

	Another challenge is collection of good data. This collection of data will require expertise from someone who is also familiar with the SONAR technology. Actual data collection will require equipment and then actual testing in ocean or another large body of water. This collection of data could not only be costly but will also require highly professional people. 

	With the help of professionals to collect good data and develop an interface for anyone to be able to use this model, this model can be used to solve problems that commercial and defense marine vessels face on every day basis. 
