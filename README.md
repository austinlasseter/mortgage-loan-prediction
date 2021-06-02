# Mortgage Loan Prediction

## Deployed app:
Application deployed on Heroku: [link](https://loan-approval-classifier-adv.herokuapp.com/)

## Problem Statement:
Dream Housing Finance company deals in home mortgage loans. They have a presence across all urban, semi urban and rural areas. The Customer first applies for a home loan after that company validates the customer eligibility for loan. The Company wants to automate the loan eligibility process in real time based on customer details provided while filling out the online application form.

## Notebook 1: `mortgage-loans-model-building.ipynb`
1. Data cleaning and exploration
2. Preprocessing the data
3. Visualizing the data
4. Exploring and optimizing various models
5. Evaluating the final model
6. Saving model components as `pickle` files

## Notebook 2: `mortgage-loans-predict-new-data.ipynb`
1. Reading in the `pickle` files
2. Writing a function to preprocess the new data
3. Create artificial data & make predictions
4. Visualize the outputs

## Final Application: `app.py` [link](https://loan-approval-classifier-adv.herokuapp.com/)
1. Deployed on Heroku
2. Receives user inputs on all features
3. Makes predictions about loan status and probability
4. Allows user to set threshold for classification
5. Visualize the applicant in comparison to the training dataset

## Data Sources
Source 1: [Kaggle](https://www.kaggle.com/ufffnick/loan-prediction-dream-housing-finance)
Source 2: [Kaggle](https://www.kaggle.com/burak3ergun/loan-data-set)


## Additional Examples:
* Linear regression model using the Ames housing dataset [here](https://ames-housing-linear-reg.herokuapp.com/).
* Logistic regression model using the home mortgage dataset [here](https://loan-approval-classifier.herokuapp.com/).
* [Mortage Loan Approval Advanced](https://loan-approval-classifier-adv.herokuapp.com/)
* [K-Nearest Neighbors with Iris Dataset](https://knn-iris-classifier.herokuapp.com/)
* [Classification with Titanic Dataset](https://titanic-classifier-2021.herokuapp.com)
* [NLP with Movie Plots](https://tmdb-rf-genres.herokuapp.com/)
* [NLP with Medical Transcripts](http://austinlasseter.com/LDA-medical-transcripts/)
* [NLP with DBPedia Entries](https://austinlasseter.medium.com/deploy-an-nlp-classification-model-with-amazon-sagemaker-and-lambda-cd5ea6339781)
* [Webscraping & NLP with Reddit Posts](https://reddit-webscraper.herokuapp.com/)
* [Great article about using SageMaker for computer vision](https://aws.amazon.com/blogs/machine-learning/deploying-machine-learning-models-as-serverless-apis/)
