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
The dataset was provided by the Kaggle competition and consists of microstructure bond trade data. The raw dataset provides 61 features and 762,678 instances. The data structure is easy to work with as there is no time series dependency, and therefore each instance is not dependent on those around it.

The first step of data preprocessing was to clean the data. We first removed irrelevant columns, namely the trade id and reporting delay features. The trade id feature was simply an identifier and had no predictive power. The reporting delay feature was removed because the it was found to be extremely noisy and did not provide any valuable insight. The next step of preprocessing was to remove all rows containing nan, infinite, and missing values. Generally, removing all rows with a corrupted value can be dangerous because it can lead to a severe reduction in the size of your data set. However, after this second step was performed the data consisted of 745,009 instances, so we are confident that this procedure was safe to perfrom and still left us with a significant amount of data to test and train.

## Model Implementation
#### Principle Component Analysis

#### Extreme Gradient Boosted Random Forest (XGBoost)

#### SVM

#### Hyperparamter Tuning


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
