# Titanic Survivor Predictor: Project Overview 
* Created a tool that predicts whether a Titanic passenger survived or not.
* Used the Train and Test datasets provided by Kaggle
* Feature engineering to the Cabin, Tickets columns
* Optimized Naive Bayes, Logistic Regression, Decision Tree, KNN, SV, XGBoost, Voting Classifiers using GridsearchCV and RandomizedSearchCV to reach the best model.
* Used the Ensembe Approach to improve the predictions

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn
**Data set:** https://www.kaggle.com/c/titanic/data

## Light Data Exploration
**For numeric data:
* Made histograms to understand distributions
* Corrplot
* Pivot table comparing survival rate across numeric variables

**For Categorical Data
* Made bar charts to understand balance of classes
* Made pivot tables to understand relationship with survival

## Feature Engineering
* Cabin - Simplify cabins (evaluated if cabin letter (cabin_adv) or the purchase of tickets across multiple cabins (cabin_multiple) impacted survival)
* Tickets - Do different ticket types impact survival rates?
* Does a person's title relate to survival rates?

## Data Preprocessing for Model
* Drop null values from Embarked (only 2)
* Include only relevant variables (Since we have limited data, I wanted to exclude things like name and passanger ID so that we could have a reasonable number of features for our models to deal with)
Variables: 'Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'cabin_adv', 'cabin_multiple', 'numeric_ticket', 'name_title'

* Do categorical transforms on all data. Usually we would use a transformer, but with this approach we can ensure that our traning and test data have the same colums. We also may be able to infer something about the shape of the test data through this method. I will stress, this is generally not recommend outside of a competition (use onehot encoder).
* Impute data with mean for fare and age (Should also experiment with median)
* Normalized fare using logarithm to give more semblance of a normal distribution
* Scaled data 0-1 with standard scaler.

## Model Building (Baseline Validation Performance)
* Naive Bayes (71.99%)
* Logistic Regression (82%)
* Decision Tree (79.08%)
* K Nearest Neighbor (81.55%)
* Random Forest (80.05%)
* Support Vector Classifier (83%)
* Xtreme Gradient Boosting (81.89%)
* Soft Voting Classifier - All Models (82.56%)

## Model Tuning
After getting the baselines, let's see if we can improve on the indivdual model results!I mainly used grid search to tune the models. 
I also used Randomized Search for the Random Forest and XG boosted model to simplify testing time.
| Model                     | Baseline | Tuned Performance  |
| ------------------------- |:--------:| ------------------:|
| Naive Bayes               | 71.99%   | NA                 |
| Logistic Regression       | 82%      | 82.6%              |
| Decision Tree             | 79.08%   | NA                 |
| K Nearest Neighbor        | 81.55%   | 83%                |
| Random Forest             | 80.05%   | 83.6%              |
| Support Vector Classifier | 83%      | 83.2%              |
| Xtreme Gradient Boosting  | 81.89%   | 85.3%              |

## Model Additional Ensemble Approaches
* Experimented with a hard voting classifier of three estimators (KNN, SVM, RF) (81.6%)

* Experimented with a soft voting classifier of three estimators (KNN, SVM, RF) (83.3%) (Best Performance)

* Experimented with soft voting on all estimators performing better than 80% except xgb (KNN, RF, LR, SVC) (82.9%)

* Experimented with soft voting on all estimators including XGB (KNN, SVM, RF, LR, XGB) (82.5%)
