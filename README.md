
# Evaluation Metrics - Lab

## Introduction

In this lab, we'll calculate various evaluation metrics to compare to evaluate classifier performance!

## Objectives

You will be able to:

* Read and interpret results using a Confusion Matrix
* Calculate and interpret precision and recall and evaluation metrics for classification
* Calculate and interpret accuracy and f1-score as evaluation metrics for classification

## Getting Started

For this lab, you're going to read in a DataFrame containing various predictions from different models, as well as the ground-truth labels for the dataset that each model was making predictions on. You'll also write various functions to help you easily calculate important evaluation metrics such as **_Precision_**, **_Recall_**, **_Accuracy_**, and **_F1-Score_**.

Let's start by reading in our dataset. You'll find the dataset stored in `'model_performance.csv'`. In the cell below, use pandas to read this dataset into a DataFrame, and inspect the head.

The dataset consists of model predictions from 3 different models, as well as the corresponding labels for row in the dataset. 

In the cell below, store each of the following predictions and labels in separate variables.


```python
model1_preds = None
model2_preds = None
model3_preds = None
labels = None
```

Good! Now, let's get started by building a confusion matrix!

## Confusion Matrix

In the cell below, complete the `conf_matrix` function.  This function should:

* Take in 2 arguments: 
    * `y_true`, an array of labels
    * `y_pred`, an array of model predictions
* Return a Confusion Matrix in the form of a dictionary, where the keys are `'TP', 'TN', 'FP', 'FN'`. 


```python
def conf_matrix(y_true, y_pred):
    pass
```

Great! Now, let's double check that our function was created correctly by creating confusion matrices for each of our 3 models. Expected outputs have been provided for you to check your results against.


```python
# Model 1 Expected Output: {'TP': 6168, 'TN': 2654, 'FP': 346, 'FN': 832}
model1_confusion_matrix = None
model1_confusion_matrix
```


```python
# Model 2 Expected Output: {'TP': 3914, 'TN': 1659, 'FP': 1341, 'FN': 3086}
model2_confusion_matrix = None
model2_confusion_matrix
```


```python
# Model 3 Expected Output: {'TP': 5505, 'TN': 2319, 'FP': 681, 'FN': 1495}
model3_confusion_matrix = None
model3_confusion_matrix
```

## Checking Our Work with sklearn

To check our work, let's make use the the `confusion_matrix()` function found in `sklearn.metrics` to create some confusion matrices and make sure that sklearn's results match up with our own.

In the cells below, import the `confusion_matrix()` function, use it to create a confusion matrix for each of our models, and then compare the results with the confusion matrices we created above. 


```python
from sklearn.metrics import confusion_matrix

model1_sk_cm = None
model1_sk_cm
```


```python
model2_sk_cm = None
model2_sk_cm
```


```python
model3_sk_cm = None
model3_sk_cm
```

## (Optional) Visualizing Confusion Matrices

In the cells below, use the visualization function shown in the **_Confusion Matrices_** lesson to visualize each of the confusion matrices created above. 

## Calculating Evaluation Metrics

Now, we'll use our newly created confusion matrices to calculate some evaluation metrics. 

As a reminder, here are the equations for each evaluation metric we'll be calculating in this lab:

### Precision

$$Precision = \frac{\text{Number of True Positives}}{\text{Number of Predicted Positives}}$$

### Recall

$$Recall = \frac{\text{Number of True Positives}}{\text{Number of Actual Total Positives}}$$

### Accuracy

$$Accuracy = \frac{\text{Number of True Positives + True Negatives}}{\text{Total Observations}}$$

### F1-Score

$$F1-Score = 2\ \frac{Precision\ x\ Recall}{Precision + Recall}$$

In each of the cells below, complete the function to calculate the appropriate evaluation metrics. Use the output to fill in the following table: 

|  Model  | Precision | Recall | Accuracy | F1-Score |
|:-------:|:---------:|:------:|:--------:|:--------:|
| Model 1 |     0.94688363524716      |    0.8811428571428571    |     0.8822     |     0.9128311380790292     |
| Model 2 |     0.744814462416746      |    0.5591428571428572    |     0.5573     |    0.6387596899224806      |
| Model 3 |    0.8899127061105723      |   0.7864285714285715     |    0.7824      |     0.8349764902168968     |

**_QUESTION:_** Which model performed the best? How do arrive at your answer?


```python
def precision(confusion_matrix):
    pass
print(precision(model1_confusion_matrix)) # Expected Output: 0.94688363524716
print(precision(model2_confusion_matrix)) # Expected Output: 0.744814462416746
print(precision(model3_confusion_matrix)) # Expected Output: 0.8899127061105723
```


```python
def recall(confusion_matrix):
    pass

print(recall(model1_confusion_matrix)) # Expected Output: 0.8811428571428571
print(recall(model2_confusion_matrix)) # Expected Output: 0.5591428571428572
print(recall(model3_confusion_matrix)) # Expected Output: 0.7864285714285715
```


```python
def accuracy(confusion_matrix):
    pass

print(accuracy(model1_confusion_matrix)) # Expected Output: 0.8822
print(accuracy(model2_confusion_matrix)) # Expected Output: 0.5573
print(accuracy(model3_confusion_matrix)) # Expected Output: 0.7824
```


```python
def f1(confusion_matrix):
    pass

print(f1(model1_confusion_matrix)) # Expected Output: 0.9128311380790292
print(f1(model2_confusion_matrix)) # Expected Output: 0.6387596899224806
print(f1(model3_confusion_matrix)) # Expected Output: 0.8349764902168968
```

Great Job! Let's check our work with sklearn. 

## Calculating Metrics with sklearn

Each of the metrics we calculated above are also available inside the `sklearn.metrics` module.  

In the cell below, import the following functions:

* `precision_score`
* `recall_score`
* `accuracy_score`
* `f1_score`

Then, use the `labels` and the predictions from each model (not the confusion matrices) to double check the performance of our functions above. 


```python
# Import everything needed here first!


preds = [model1_preds, model2_preds, model3_preds]

for ind, i in enumerate(preds):
    print('-'*40)
    print('Model {} Metrics:'.format(ind + 1))
    print('Precision: {}'.format(None))
    print('Recall: {}'.format(None))
    print('Accuracy: {}'.format(None))
    print('F1-Score: {}'.format(None))
```

## Classification Reports

Remember that table that you filled out above? It's called a **_Classification Report_**, and it turns out that sklearn can even create one of those for you! This classification report even breaks down performance by individual class predictions for your model. 

In closing, let's create some and interpret some classification reports using sklearn. Like everything else we've used this lab, you can find the `classification_report()` function inside the `sklearn.metrics` module.  This function takes in two required arguments: labels, and predictions. 

Complete the code in the cell below to create classification reports for each of our models. 


```python
# Import classification_report below!

for ind, i in enumerate(preds):
    print('-'*40)
    print("Model {} Classification Report:".format(ind + 1))
    print(None)
```

## Summary

In this lab, we manually calculated various evaluation metrics to help us evaluate classifier performance, and we also made use of preexisting tools inside of sklearn for the same purpose. 
