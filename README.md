# deep-learning-challenge
Module 21 - Machine Learning and Neural Networks - May 2023

# Report on the Neural Network Model

## Overview of the analysis
The goal of the analysis is to develop a tool that will help the nonprofit foundation Alphabet Soup to select the applicants for funding with the best chance of success in their ventures. Using the features in their dataset of more than 34,000 organizations that have received funding from Alphabet Soup over the years, I created a binary classifier to predict whether applicants will be successful if funded by Alphabet Soup.

## Results 

### Data Preprocessing

The target variable of the model is "Is_Successful" as that is what we want to predict.

The features of the model are the remaining columns that were not dropped: APPLICATION_TYPE,	AFFILIATION,	CLASSIFICATION,	USE_CASE, ORGANIZATION,	STATUS,	INCOME_AMT,	SPECIAL_CONSIDERATIONS and ASK_AMT.

The identification variables EIN and NAME were removed from the input data because they are neither targets nor features. 

### Compiling, Training, and Evaluating the Model

Initially, I tried using 2 hidden layers with 80 and 30 neurons and using the sigmoid and renu activation functions. The output layer used the tanh activation function. After 100 epachs, the training accuracy went up to 0.7395 and the test accuracy was 0.7324. However, the target model performance was 0.75.

To increase model performance, I tried removing additional columns which seemed to be of minor significance - STATUS and SPECIAL_CONSIDERATIONS. I also tried additional binning and increasing the number of layers, neurons epochs as well as changing the activation functions. The optimized model had a training accuracy of 0.7388 and test accuracy of 0.7331. 

### Summary
In summary, through multiple iterations, a model was created that did not achieve the desired accuracy of 0.75 but got to 0.733. An SVM binary classification model may also be used to solve this problem as we are dealing with two options for the target - is successful and is not successful.

![App](images/Screen%20Shot%202023-05-29%20at%207.36.32%20PM.png)
![App](images/Screen%20Shot%202023-05-29%20at%207.38.00%20PM.png)











# Assignment Details
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

## Instructions
Step 1: Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.
Use any or all of the following methods to optimize your model:
- Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
  - Dropping more or fewer columns.
  - Creating more bins for rare occurrences in columns.
  - Increasing or decreasing the number of values for each bin.
- Add more neurons to a hidden layer.
- Add more hidden layers.
- Use different activation functions for the hidden layers.
- Add or reduce the number of epochs to the training regimen.
Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.
Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.
The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.

Results: Using bulleted lists and images to support your answers, address the following questions:
  Data Preprocessing
  - What variable(s) are the target(s) for your model?
  - What variable(s) are the features for your model?
  - What variable(s) should be removed from the input data because they are neither targets nor features?

Compiling, Training, and Evaluating the Model
- How many neurons, layers, and activation functions did you select for your neural network model, and why?
- Were you able to achieve the target model performance?
- What steps did you take in your attempts to increase model performance?

Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
