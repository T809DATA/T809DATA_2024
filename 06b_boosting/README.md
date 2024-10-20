# Boosting

In this exercise we'll use boosting on the Titanic dataset from [Kaggle](kaggle.com). The dataset, demos and discussions on the dataset can be found [here](https://www.kaggle.com/c/titanic). The raw titanic data can be accessed via `tools.get_titanic`

To use boosting, we will use the `GradientBoostingClassifier` in `sklearn`.

We will be using the pandas library to organize and clean up the data. The data we're given here has quite a few defects (which is quite common when dealing with real data) and we need to fix and clean it up before we do our predictions. A cheat-sheet for pandas can be found [here](http://datacamp-community-prod.s3.amazonaws.com/dbed353d-2757-4617-8206-8767ab379ab3)

**Note**: You might have to update your requirements by running `pip install -r requirements.txt` from the root assignment directory.

**About the Titanic Dataset**:
This dataset contains information about passengers aboard the Titanic. It also contains information about which passengers survived the shipwreck and which didn't. This dataset is often used to train models to predict from passenger information which passengers survived and which didn't.

For each passenger, at least some of the following information is given:

| Variable | Definition | Key |
| ---| --- | --- |
| survival | Survival | 0 = No, 1 = Yes |
| pclass | Ticket class |1 = 1st, 2 = 2nd, 3 = 3rd |
| sex | Sex | - |
| Age | Age in years | - |
| sibsp | # of siblings / spouses aboard the Titanic | - |
| parch | # of parents / children aboard the Titanic | - |
| ticket | Ticket number | - |
| fare | Passenger fare | - |
| cabin | Cabin number | - |
| embarked | Port of Embarkation| C = Cherbourg Q = Queenstown, S = Southampton |


## Section 1
We have to clean up the data. We have provided you with the function `tools.get_titanic` that cleans up the data and returns you : `(X_train, y_train), (X_test, y_test), submission_X`:
* `X_train`: The features for the training set
* `t_train`: The targets for the training set
* `X_test`: The features for the test set
* `t_test`: The targets for the test set
* `submission_X`: Features that you will use to make predictions about the survival of passengers without any knowledge of the corresponding `submission_y`.

Example usage:
```
>>> (tr_X, tr_y), (tst_X, tst_y), submission_X = get_titanic()
# get the first 1 row in the training features
>>> tr_X[:1])

     Pclass  SibSp  Parch    Fare  Sex_male  Cabin_mapped_1  Cabin_mapped_2  ...  Cabin_mapped_4  Cabin_mapped_5  Cabin_mapped_6  Cabin_mapped_7  Cabin_mapped_8  Embarked_Q  Embarked_S
794       3      0      0  7.8958         1               0               0  ...               0               0               0               0               0           0           1

[1 rows x 15 columns]

```
Notice that all these sets are **not** `numpy` arrays but `pandas` data frames.


### Section 1.1
Take a look at `tools.get_titanic`. In the middle of the function we drop a couple of columns that contain NaNs (not numbers):
```
X_full.drop(
    ['PassengerId', 'Cabin', 'Age', 'Name', 'Ticket'],
    inplace=True, axis=1)
```

You can take a look at the data frame before performing the drop with e.g. `print(X_full[:10])` and you should see the first 10 rows with these age values:
```
22.0
38.0
26.0
35.0
35.0
NaN
54.0
2.0
27.0
14.0
```

But maybe we can replace the `NaN` values with some other information? Take a look at `get_titanic` for clues.

Make a new function in your `template.py` called `get_better_titanic` that does this.

### Section 1.2

Write a short summary about your change to `get_titanic`.

## Section 2
We will now try training a random forest classifier on the titanic data.

### Section 2.1
Create a function `rfc_train_test(X_train, t_train, X_test, t_test)` that trains a Random Forest Classifier on `(X_train, t_train)` and returns:
* accuracy
* precision
* recall

on `(X_test, t_test)`

Example inputs and outputs:
```
>>> rfc_train_test(tr_X, tr_y, tst_X, tst_y)
(0.8097014925373134, 0.7708333333333334, 0.7184466019417476)
```

### Section 2.2

Upload your choice of parameters and the accuracy, precision and recall on the test set.

### Section 2.3
Create a function `gb_train_test(X_train, t_train, X_test, t_test)` that trains a `GradientBoostingClassifier` on `(X_train, t_train)` and returns:
* accuracy
* precision
* recall

on `(X_test, t_test)`

Example inputs and outputs:
```
>>> gb_train_test(tr_X, tr_y, tst_X, tst_y)
(0.8208955223880597, 0.8313253012048193, 0.6699029126213593)
```

### Section 2.4

Upload the Gradient boosting classifier accuracy, precision and recall on the test set. How does it compare to the random forest classifier?

### Section 2.5

We will now try to perform [randomized parameter search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Random_search) to find better parameters for the gradient boosting classifier. We will be using the `randomizedSearchCV` import to help us do this.

Fill in the blanks in the `param_search` function:
* `n_estimators`: Choose multiple integer values between 0 and 100
* `max_depth`: Choose multiple integer values no more than 50.
*  `learning_rate`: Choose multiple float values between 0 and 1.

By calling the function with the training features and training targets, it will perform the parameter search and return the best value of `n_estimators`, `max_depth` and `learning_rate` it can find.

Example usages:
```
>>> param_search(tr_X, tr_y)
(10, 2, 0.5)
```

Run this function and report which are the best values of parameters.


### Section 2.6
Create a function `gb_optimized_train_test` that does the same as your `gb_train_test` but now uses your optimized parameters instead.

```
>>> gb_optimized_train_test(tr_X, tr_y, tst_X, tst_y)
(0.8171641791044776, 0.7934782608695652, 0.7087378640776699)
```

**To get full marks, each score has to be higher than the one from your vanilla GB classifier**


### Section 3
Now that you have your optimized classifier we will submit it to kaggle.

### Section 3.1
Fill in the `_create_submission` function. Run the function and a `.csv` submission file will appear under `./data`.

### Section 3.2

Log in to Kaggle, join the Titanic competition and submit your `.csv` file. After the upload, kaggle will calculate your score. It should look something like:

`7132 YourName novice tier 0.77033 1 3m`

Upload this information.


## Bonus section
There are many other types of models that we didn't try on this dataset that might work better. Maybe there are better parameter optimizations that could be done. Maybe the `get_titanic` function could be further improved?

You decide what you do but the goal with the bonus section is to improve your kaggle score the most.

Upload any code that you generated as part of the bonus assignment and a short summary of your experimentation, solution and results.