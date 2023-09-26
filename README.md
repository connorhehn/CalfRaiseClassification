# Calf Raise Classification
## By Connor Hehn and John Minogue

### Table of Contents
-[Introduction](#introduction)
-[Related Works](#related-works)
-[Data](#data)
-[Features](#features)
-[Methods/Models](#methods/models)
-[Evaluatoin](#evaluation)
-[Conclusion](#conclusion)
-[References](#references)

### Introduction
The Fairfield University Community Situated Biomechanics Lab was just recently established in September of 2022. The main focus of the lab is centered around tracking musculoskeletal movements of athletes among a very wide age range. The lab’s equipment, including a motion capturing system and plantar force sensing load sols (shoe sols), needs to be synced together and validated in this new space in order for future human subject testing to take place. This equipment will be used to collect data on ankle kinematics and plantar force while 10 subjects perform a calf raise either from the ground, or with their foot elevated on a box. With this data, machine learning will be utilized to create a model that will predict which of the two aforementioned calf raise methods is being performed. This experiment is meant to validate the collaborative work between our lab equipment and our machine learning model. Once this system has been validated, future achilles tendon rupturing research regarding causation and preventative procedures will immediately follow.  
### Related Works
Novel USA Load sols and xGen Motion Monitor Software are fairly new technologies with limited experiments and papers written about them. There are two interesting related studies and papers regarding the use of the USA Load Sols. The first paper conducts research on the validity and reliability of the Load Sols technology under various walking and running conditions[4]. The experiment consisted of 30 people and tested each individual walking and running at 0%, 10% and -10% inclines. The paper concluded that the Load Sols had excellent between day reliability measured at greater than 0.76. This experiment is similar to our project in determining the validity of the equipment, although our experiment does not utilize running and walking at different speeds and inclines, rather our project utilizes and measures a calf raise maneuver. Our project also incorporates the xGen Motion Monitor Software, which this particular paper does not examine.
Another interesting study utilizes machine learning to predict a runner's foot strike angle when wearing the Load Sol technology[3]. The paper utilizes three machine learning techniques, “multiple linear regression―MR, conditional inference tree―TREE, and random forest―FRST”[3]. This is related to our project in that this study utilized machine learning to predict outcomes when wearing the Load Sol devices. This study was able to achieve an accuracy score of greater than 90%[3]. Similar to the first study, this study examines a subject's data when they are running, or walking, whereas our paper examines stationary movement such as calf raises. Both these studies are interesting and enlightening in the area of biomechanics. Our project is similar to both of these papers in the sense that we examine data from subjects wearing the Load Sol technology, and validate the equipment. Our project not only uses Load Sol technology, but also uses xGen Motion Monitor Software whereas these studies do not.
### Data
10 subjects performed double leg calf raises from the floor and then from an elevated position to evaluate their load distribution and ankle kinematics. The subjects’ shoes contain the Novel USA Load Sols, which measure plantar force across the subject’s foot during dynamic movements. The subject will also have motion capturing markers attached to his/her thighs, calves (shank), and shoes. The markers will be tracked using the xGen Motion Monitor Software. This equipment will provide us with a multitude of data on the subject while he/she is performing calf raises. Using this data to train our model, the machine learning algorithm will then be able to predict whether the subject is doing the calf raises from the floor, or from the elevated position. 
Before the experiments could be performed, the raw data needed to be transformed to usable data. The raw data from the experiment performed at the Fairfield University Community Situated Biomechanics Lab is tracked over a 5 second period in 0.01 second intervals. Each dataset has 500 rows, and each row has 8 observed values. Therefore each datapoint, which is linked to one target value (grounded or elevated), has 4000 attributes. Because of the large number of attributes, feature engineering was performed on the data. Feature engineering is creating features based on existing ones. For each of the observed values (columns) in the raw data, the statistics were determined. The specific statistics used are average value, minimum value, maximum value, and standard deviation. These statistics are applied to every column in the raw data, and new features are created. Each column now has 4 resulting columns, with one value attached to it. The result of feature engineering changes the datapoint containing 4000 attributes into a more workable datapoint containing only 28 attributes (column “Time” was omitted because it was not unique). The following table shows the features used for the experiments conducted:
### Features
Forefoot Force[N] Avg

Forefoot Force[N] Min

Forefoot Force[N] Max

Forefoot Force[N] Std

Midfoot Force[N] Avg

Midfoot Force[N] Min

Midfoot Force[N] Max

Midfoot Force[N] Std

Heel Force[N] Avg

Heel Force[N] Min

Heel Force[N] Max

Heel Force[N] Std

TForce[N] Avg

TForce[N] Min

TForce[N] Max

TForce[N] Std

RAnkleFlexion Avg

RAnkleFlexion Min

RAnkleFlexion Max

RAnkleFlexion Std

RAnkleInversion Avg

RAnkleInversion Min

RAnkleInversion Max

RAnkleInversion Std

RAnkleAbduction Avg

RAnkleAbduction  Min

RAnkleAbduction Max

RAnkleAbduction Std

Person #

Grounded (Target)

In addition to the feature engineering, the data needed to be scaled and interpolated. Scaling was needed because ankle kinematics were measured in degrees and plantar force was measured in Newtons. While plantar force ranged between 0 and 700 N of force, ankle flexion, inversion, and abduction only ranged from -35 to 15 degrees. Plantar force would have therefore had a greater influence on the data if everything was not scaled. Interpolation was needed because the motion capturing markers occasionally were lost by the motion capturing cameras during data collection. Therefore, there were some null values while the subject was performing calf raises. 
### Methods/Models
The intended experiment will use motion capture and plantar force sensing load sols to predict whether a subject is performing a calf raise from the ground, or on an elevated surface. The collection of data will be used to predict one of two options, thus making it a binary classification. A binary classification has many possible machine learning methods including: Logistic Regression, K Nearest Neighbor, Support Vector Machines, Random Forest, and Neural Networks. For our experiment we chose to perform the ensemble method Random Forest, as well as Logistic Regression. 
#### Random Forest
For the first experiment, the chosen machine learning model was the ensemble method Random Forest. Random Forest is a collection of random tree models. Random tree models are sensitive to small changes in the dataset and have the potential to overfit the model to the training set. Random Forests are not as sensitive to small changes in the dataset, and are less likely to be overfitted to the training data. Using random forest in conjunction with leave one out cross validation will give us an idea of how well the equipment works.
#### Logistic Regression
Logistic regression models are easy to implement and understand. This kind of model is commonly used with binary classification problems. The graph on the right portrays classification done with logistic regression with only one feature/attribute (independent variable). When given a particular feature, x, the model will predict whether the target is equal to a 1 or a 0 using a sigmoid function. Our project has multiple features, so it’s harder to represent in an understandable graph, but the main function of the analysis is the same. We will be utilizing multiple logistic regression, which is specifically made for data with multiple features and only two targets. Our logistic regression model will identify patterns and similarities throughout our data to classify the calf raise as a “1”(grounded) or “0” (elevated).  
### Evaluation
Since the classification was dealing with subject data, leave one out cross validation was used to ensure the integrity of the model. With leave one out cross validation, a singular test set is represented as the entirety of one subject's data. The rest of the subjects are used for the training data. Since one subject has multiple data points, this ensures that one subject's data is not in both the training and test set. The results from the experiment convey more meaning using leave one out cross validation, because the model determines how well a new person's data is predicted when a new person is introduced. Our main performance metrics included accuracy score, f1 score, and confusion matrices. 
#### Random Forest
Initial results from the random forest algorithm were promising, with an initial accuracy of 100%. The accuracy was determined as an average accuracy. Each of the 10 tests (1 for each subject) returned an accuracy, and the average accuracy was determined as the average of the 10 tests. To further evaluate the model, the number of trees parameter was tuned in order to see what was going on in the algorithm. The default number of trees in the random forest algorithm is 100 trees. The algorithm was tuned to change the number of trees, and record the average accuracy of the test for each number of trees starting from 1 to 100. It became clear that as the number of trees increased, the accuracy increased. The accuracy was plotted against the number of trees shown below. The results show that 100 trees are not needed, rather an accuracy of 100% is achieved around 16 trees. 

The results of the random forest indicate that the equipment used produces meaningful results. The model created is able to predict at 100% accuracy the classification of an unseen person’s data whether they are performing calf raises on the ground or at an elevated position. Although the Random Forest method produces accurate results, the method is a “black box” method, meaning that there is little knowledge of the internal workings of the method. For purely accurate results, this method achieves the desired goal. For a deeper understanding of how the data predicts the result, logistic regression was utilized as another method, where we could get a closer look at the internal workings of the predictions.
#### Logistic Regression
We found an enormous amount of success utilizing the random forest model. However, we wanted to continue to validate our model by trying other methods. Additionally, random forest is a “black box” method. It’s hard to explain to end users how it works and why it is making the decisions it’s making. Logistic regression can be mathematically and graphically explained. So, we decided that as long as we can get a high performing model using logistic regression, we would abandon our random forest method for this particular project.
Using leave one out cross validation, our model found 100% accuracy for all subjects except for subject 1. Overall accuracy was 95%. We utilized scaling and SelectKBest, which tells us the most influential features in the dataset, to train our model. We found the most influential features to be average heel force and average rankle flexion. Their logistic regression curves are shown in the two figures on the right. Note that right ankle flexion was a negative value, but had to be made positive in order to run a proper logistic regression analysis on its data. The respective logistic regression functions for average heel force and right ankle flexion were  Y = 0.2992𝑒^(−𝑥) - 0.3645 and Y = 0.9228𝑒^(−𝑥) - 1.5568. The values show respective cutoffs of 30-40 Newtons of heel force and (-10)-(-12) degrees of ankle flexion. These graphs, particularly the average heel force, explain the reason why our model was not working with 100% accuracy. The average heel force graph shows 2 values that were classified not following the logistic regression curve. It turns out that these two values both belonged to subject 1, and they were actually misidentified. This tells us that subject 1 was an outlier in the dataset, which threw off our model. It’s difficult to say whether there are more people like subject 1 or if he truly is an outlier. This is because our dataset only contains 10 subjects. In order to test this, we would need to increase the number of subjects in our experiment. 
The logistic regression model was successful in reaching 95% accuracy, so it was an acceptable model to use for this particular project. However, if we wanted to guarantee 100% accuracy, we would stick with the random forest method. Random forest, although difficult to explicitly explain to others, is a complex ensemble method that works the best for this particular dataset.
### Conclusion
The Fairfield University Community Situated Biomechanics Lab’s motion capturing system and plantar force sensing load sols (shoe sols) have been synced together and validated to allow for future human subject testing to take place. Future achilles tendon rupturing research regarding causation and preventative procedures using this equipment will proceed this project. Using the data we used to validate the lab’s equipment, we successfully developed 2 machine learning models. These random forest and logistic regression models were both successful as they respectively yielded 100% and 95% accuracy. In order to improve the accuracy of the logistic regression model, a larger dataset should be collected so that one subject doesn’t have an overwhelming influence on the model. The random forest method was perfect, but a larger dataset may shed light on other problems that the model may face. While random forest was the best performing model, we stuck with the logistic regression model for the convenience of being able to explain our model and why it makes the decisions it makes. 
### References
Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.

J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007. DOI: 10.1109/MCSE.2007.55

Moore, S.R.; Kranzinger, C.; Fritz, J.; Stӧggl, T.; Krӧll, J.; Schwameder, H. Foot Strike Angle Prediction and Pattern Classification Using LoadsolTM Wearable Sensors: A Comparison of Machine Learning Techniques. Sensors 2020, 20, 6737. https://doi.org/10.3390/s20236737

Renner, K.E.; Williams, D.B.; Queen, R.M. The Reliability and Validity of the Loadsol® under Various Walking and Running Conditions. Sensors 2019, 19, 265. https://doi.org/10.3390/s19020265

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
The pandas development team. pandas-dev/pandas: Pandas. Zenodo. February 2020. DOI: 10.5281/zenodo.3509134 

Python Libraries (References Above)
pandas
numpy
matplotlib.pyplot
from sklearn:
(.ensemble) RandomForestClassifier
(.feature_selection) f_classif
(.feature_selection) SelectKBest
(.linear_model) LogisticRegression
(.metrics) accuracy_score
(.metrics) confusion_matrix
(.metrics) ConfusionMatrixDisplay
(.metrics) f1_score
(.preprocessing) StandardScaler