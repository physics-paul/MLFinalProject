## Meet the Team
Group 20
- Carson Fletcher
- Sidney Genrich
- Clemens Koolen 
- Andrew Palmer
- Ben Wisehaupt

## Introduction
Our project is based on a kaggle competition to predict the next price that a US corporate bond might trade at. The movement of asset prices is a highly complex system, making them a key candidate for machine learning techniques. Given the various new platforms on which corporate bonds are traded, and the need for more frequent, up-to-date information of bond prices, we hope to create an algorithm that is able to accurately, and quickly price bonds.

## Data
The dataset was provided by the Kaggle competition and consists of microstructure bond trade data. The raw dataset provides a label coumn decsribing the true trade price, a weight column, 57 features, and 762,678 instances. The weight column is  calculated as the square root of the time since the last trade and then scaled so the mean is 1. This weight column is used for evaluation purposes and will be further discussed below. The data structure is easy to work with as there is no time series dependency, and therefore each instance is not dependent on those around it.

The first step of data preprocessing was to clean the data. We first removed irrelevant columns, namely the trade id and reporting delay features. The trade id feature was simply an identifier and had no predictive power. The reporting delay feature was removed because the it was found to be extremely noisy and did not provide any valuable insight. The next step of preprocessing was to remove all rows containing nan, infinite, and missing values. Generally, removing all rows with a corrupted value can be dangerous because it can lead to a severe reduction in the size of your data set. However, after this second step was performed the data consisted of 745,009 instances, so we are confident that this procedure was safe to perfrom and still left us with a significant amount of data to test and train.

Lastly, the data was split into a testing and training set. Because the data does not have any timeseries dependency we were able to use the sklearn test_train_split function to randomly split the data.  Because our data set was plentiful we chose to use a split size of 75% training and 25% testing, which resulted in a training size of 558,756 instances and a testing size of 186,253 instances. 

Functionality was added to scale the data between zero and one, and normalize the data with a mean of zero and standard deviation of 1. These functions allow us to observe the effectiveness of standardizing the data prior to model implementation and then choose the option that provides the best performance.

## Model Implementation

In order to choose our models, we have to first make some observations about our data. The most important observation for our model selection is that unlike many other datasets out there that might classify things, or regress seemingly linear data, our data is highly non-linear. This severely limits our scope of machine learning models. As a result, for our supervised learning algorithms, we will be looking at three different levels of decision trees in the form of a simple decision tree, a random forest, and an extreme gradient boosted random forest. We beleive these models will be able to effectively solve our problem because they perform well when using non-linear data. This pivot to new supervised models is the biggest change in our approach from the approach presented in the project proposal.

#### Principle Component Analysis
Our unsupervised learning model is a principal component analysis which was used to reduce the dimensionality of the data set. Given the complexity of our machine learning models, reducing the number of dimensions is extremely valuable because it reduces the amount of time needed to fit the models.




#### Decision Tree Regressor



#### Random Forest Regressor

As our second Supervised model, we have chosen a Random Forest Regressor model, as sort of an improvement on our Decision Tree Regressor. Given the large amount of variables and data points, and the very specific objective we are trying to achieve, it is important that we get a large amount of accuracy, without risking overfitting, which we are likely to encounter in a decision tree. If we look at how a Decision Tree Regressor is fitted on data, we don't really create a smart algorithm. Our model will see a new data point and simply compare the new point with what it has seen previously, regardless of bias or specific important parameters in the data. Therefore, to avoid all of this, one of the most obvious models to use is a random forest. Using the characteristic bootstrapping that a Random Forest employs, we can get rid of any bias in our data. Data points will be trained individually given a sample of features, and through this process, we will aggregate the results of various decision trees that have all been trained on different kinds of data and features. As a result, we have taken the benefits of our decision tree, and combined it with unbiased data and smarter regression.

For implementation

For hyperparameter tuning, we will look at both parameters that are characteristic to decision trees, and to random forests. For example, one of the parameters specific to the random forest is the number of trees. Naturally, the more trees we have the better our model will perform, yet at the cost of speed. Similarly, we can look at the decision trees in the forest, and recognize that using our regression, we would want to change the maximum number of leaves of a node, or ... **insert parameters**

For the implementation of the parameter tuning, we first recognized that there is a large variability in our data and features. We therefore won't just run a simple GridSearch with arbitrary hyperparameters, as we could either make the model too complicated, or too simple. Therefore, to give us an idea of what we should put in our GridSearch, we will first run a RandomSearch, looking at a wide range of hyperparameters and see how it impacts our results. **Insert ranges**



#### Extreme Gradient Boosted Random Forest (XGBoost)



## Performance Evalaution

## Conclusions





You can use the [editor on GitHub](https://github.com/cfletcher33/cfletcher33.github.io/edit/master/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/cfletcher33/cfletcher33.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
