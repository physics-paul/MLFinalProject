
# =============================================================================
# Imports
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV

# =============================================================================
# Plot styling
# =============================================================================

sns.set()
plt.style.use('seaborn-darkgrid')
matplotlib.rcParams['figure.figsize'] = (8,8)

SMALL_SIZE = 13
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


matplotlib.rcParams.update({'font.weight': 'bold'})
matplotlib.rcParams.update({'figure.titleweight': 'bold'})
matplotlib.rcParams.update({'axes.labelweight': 'bold'})
matplotlib.rcParams.update({'axes.titleweight': 'bold'})

matplotlib.rcParams.update({'legend.frameon': True})
matplotlib.rcParams["legend.facecolor"] = 'white'
matplotlib.rcParams["legend.framealpha"] = 1

# =============================================================================
# =============================================================================
# # Import Data
# =============================================================================
# =============================================================================

#load the raw data
print("Loading in raw data...")
raw_data = pd.read_csv('/Users/andrewpalmer/Desktop/QCF_Spring_2020/Machine_Learning/ML-Final-Project/data/raw_data.csv')

# =============================================================================
# =============================================================================
# # Define functions
# =============================================================================
# =============================================================================

#function to clean and split the data
def clean_data(raw_data):
    """
    This function takes in the rawest form of the data set after CSV import and
    keeps only predicting and predicted columns. Rows with missing data are dropped.
    Lastly, the data is split into testing and training sets.
    """
    #remove irrelevant columns
    data = raw_data.drop(columns=['id', 'bond_id', 'reporting_delay'])
    
    #remove NaN by row
    data = data.dropna().reset_index(drop=True)
    
    #separate into X and y data
    x = data.drop(columns='trade_price')
    y = data[['trade_price']]
    
    #split into testing and training data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
    
    # extract weights
    train_weights = x_train[['weight']]
    test_weights = x_test[['weight']]
    
    # reset indices and remove weights
    x_train = x_train.reset_index(drop=True).drop(columns='weight')
    x_test = x_test.reset_index(drop=True).drop(columns='weight')
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return x_train, x_test, y_train, y_test, train_weights, test_weights

#function to scale the columns between 0 and 1
def scale_data(x_data):
    """
    Parameters:
        x_data - the data that will be min-max scaled
    Returns:
        scaled_data -  the passed-in dataset scaled column-wise between the min and max
    """
    #get the column names
    target_cols = x_data.columns
    
    #perform scaling on the features
    scaler = MinMaxScaler()
    scaled_data = x_data.copy()
    scaled_data[target_cols] = scaler.fit_transform(x_data[target_cols])
    
    return scaled_data

#function to normalize the columns with mean=0 and stdev=1
def normalize_data(x_data):
    """
    Parameters:
        x_data - the data that will be normalized with mean 0 and stdv 1
    Returns:
        scaled_data -  the passed-in dataset normalized column-wise
    """
    #get the column names
    target_cols = x_data.columns
    
    #perform normalization
    normalizer = StandardScaler()
    normalized_data = x_data.copy()
    normalized_data[target_cols] = normalizer.fit_transform(x_data[target_cols])
    
    return normalized_data

# Returns the number of intrinsic dimensions of the dataset in accordance with threshold
def n_components_pca(explained_variance_ratios,threshold=0.97):
    """
    Parameters:
        explained_variance_ratios - a vector of percent variance explained by each 
            principal component
        threshold - the percent of total variance to consider a fidelitous representation
            of original data
    Returns:
        Number of principal components that capture threshold of variance
    """
    cumsum=0
    for i in range(len(explained_variance_ratios)):
        cumsum += explained_variance_ratios[i]
        if cumsum>threshold:
            return i+1  

#fuction to perform PCA on the features, requires access to scale_data function
def pca(x_train_data, x_test_data, threshold = 0.97):
    """
    Parameters:
        x_train_data - x training data where PCA will be run for intrinsic dimension analysis
        x_test_data - x testing data where PCA results from training will be applied
    Return:
        PCA_data_train - PCA transormed and reduced x training data
        PCA_data_test - PCA transormed and reduced x testing data
    """
    #scale the data
    x_train_scaled = normalize_data(x_train_data)
    x_test_scaled = normalize_data(x_test_data)
    
    #get the column names
    target_cols = x_train_data.columns
    
    #perfrom PCA on the features
    pca = PCA()
    principle_components_train = pca.fit_transform(x_train_scaled[target_cols])
    principle_components_test = pca.transform(x_test_scaled[target_cols])
    
    # Find intrinsic dimensions and visualize
    explained_var_ratio = pd.Series(pca.explained_variance_ratio_)
    n_true_components = n_components_pca(explained_var_ratio,threshold)
    print("True component count:\t",n_true_components)
    
    # Explained Var Ratio Plot
    explained_var_ratio.plot(kind='bar',
                             figsize=(10,10),
                             label="Principal Component % Variance Captured",
                             color='r')
    plt.vlines(n_true_components-0.5,
               linestyles=[(0,(9,3,4,4))], 
               ymin=0, 
               ymax=max(explained_var_ratio), 
               alpha = 0.5, 
               label="Number of Components to Caputure %.1f%% variance"%(threshold*100))
    plt.title('Explained Variance Ratios For Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel("Principal Component")
    legend = plt.legend()
    legend.get_frame().set_facecolor('White')
    plt.savefig("PCA.png")
    plt.show()
    
    #save the principle components
    PCA_columns = []
    for i in range(0,n_true_components):
        PCA_columns.append('pc_'+str(i+1))
        
    PCA_data_train = pd.DataFrame(data=principle_components_train[:,:n_true_components], columns=PCA_columns)
    PCA_data_test = pd.DataFrame(data=principle_components_test[:,:n_true_components], columns=PCA_columns)
    return PCA_data_train, PCA_data_test


def Gradient_Boosted_fit(model, X_train, Y_train, train_weights):
    # Build data set
    xgtrain = xgb.DMatrix(X_train,
                          label=np.array(Y_train),
                          weight=np.array(train_weights),
                          feature_names=X_train.columns)
    
    # Cross Validation
    print("XGBoost Cross Validation...")
    cvresult = xgb.cv(model.get_xgb_params(), 
                      xgtrain, num_boost_round=model.get_params()['n_estimators'], 
                      nfold=5,
                      metrics={'mae'}, 
                      early_stopping_rounds=5,
                      verbose_eval=True)
    print("CV for n-estimators resulted in %d estimators"%(cvresult.shape[0]))
    model.set_params(n_estimators=cvresult.shape[0])
    
    # CV Plot
    plt.plot(cvresult[['train-mae-mean']],'b',label="Train Mean MAE")
    plt.plot(cvresult[['test-mae-mean']],'r--',label="Test Mean MAE")
    plt.legend()
    plt.xlabel("N Estimators")
    plt.ylabel("MAE")
    plt.ylim(0,2)
    plt.title("Cross Validation for Number of Estimator Results",fontsize=20)
    plt.show()

    
    # Refit and evaluate
    print("Training Set Results")
    model.fit(np.array(X_train), np.array(Y_train),eval_metric='mae')
    y_train_pred = model.predict(np.array(X_train))
    print("Training MAE:\t",mean_absolute_error(Y_train,y_train_pred,train_weights))
    
    # Feature Importances plot for training data
    feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(20).plot(kind='barh',color='r')
    plt.title("Feature Importances",fontsize=20)
    plt.xlabel("F Score")
    plt.show()
    
    return model

def model_evaluation(labels,prediction,evaluation_weights,x_test):
    y_x_test = x_test.copy()
    y_x_test['Prediction'] = prediction
    y_x_test['Actual'] = labels
    y_x_test = y_x_test[['Actual','Prediction','trade_price_last1']]
    y_x_test['LongPosition'] = y_x_test['trade_price_last1'].lt(y_x_test['Prediction'])
    y_x_test['LongProfit'] = y_x_test['Actual'] - y_x_test['trade_price_last1']
    y_x_test['ShortProfit'] = y_x_test['trade_price_last1'] - y_x_test['Actual']
    y_x_test['LongReturn'] = y_x_test['LongProfit'] / y_x_test['trade_price_last1']
    y_x_test['ShortReturn'] = y_x_test['ShortProfit'] / y_x_test['trade_price_last1']
    y_x_test['ModelProfit'] = np.where(y_x_test['LongPosition'].values, y_x_test['LongProfit'], y_x_test['ShortProfit'])
    y_x_test['ModelReturn'] = np.where(y_x_test['LongPosition'].values, y_x_test['LongReturn'], y_x_test['ShortReturn'])
    
    y_x_test['CumulativeLongProfit'] = np.cumsum(np.where(y_x_test['LongPosition'].values, y_x_test['LongProfit'], 0))
    y_x_test['CumulativeLongReturn'] = np.cumprod(1+np.where(y_x_test['LongPosition'].values, y_x_test['LongReturn'], 0))-1
    
    y_x_test['CumulativeShortProfit'] = np.cumsum(np.where(y_x_test['LongPosition'].values, 0, y_x_test['ShortProfit']))
    y_x_test['CumulativeShortReturn'] = np.cumprod(1+np.where(y_x_test['LongPosition'].values, 0, y_x_test['ShortReturn']))-1
    
    y_x_test['CumulativeModelProfit'] = np.cumsum(y_x_test['ModelProfit'])
    y_x_test['CumulativeModelReturn'] = np.cumprod(1+y_x_test['ModelReturn'])-1
    
    long_profit = y_x_test['CumulativeLongProfit'].iloc[-1]
    long_return = y_x_test['CumulativeLongReturn'].iloc[-1]
    
    short_profit = y_x_test['CumulativeShortProfit'].iloc[-1]
    short_return = y_x_test['CumulativeShortReturn'].iloc[-1]
    
    total_profit = y_x_test['CumulativeModelProfit'].iloc[-1]
    total_return = y_x_test['CumulativeModelReturn'].iloc[-1]
    
    # Report metrics
    wmae = mean_absolute_error(labels,prediction,evaluation_weights)
    
    print('Long Profit:\t',long_profit)
    print('Long Return:\t',long_return)
    print()
    print('Short Profit:\t',short_profit)
    print('Short Return:\t',short_return)
    print()
    print('Total Profit:\t',total_profit)
    print('Total Return:\t',total_return)
    print()
    print("Testing WMAE:\t",wmae)
    
    return wmae

# Decision Tree
def decision_tree(x_train, y_train, train_weights):
    
    dt = tree.DecisionTreeRegressor()

    #define parameters
    criterion = ['mse']
    splitter = ['best', 'random']
    max_depth = [int(x) for x in range(40, 75, 5)]
    max_depth.append(None)
    max_features = ['auto', 'sqrt']
    min_samples_leaf = [1, 2, 4]
    min_samples_split = [2, 5, 10]

    # Create the random grid
    random_grid = {'criterion': criterion,
                    'splitter': splitter,
                    'max_depth': max_depth,
                    'max_features': max_features,
                    'min_samples_leaf': min_samples_leaf,
                    'min_samples_split': min_samples_split}
    
    model = RandomizedSearchCV( estimator = dt, 
                                param_distributions = random_grid, 
                                cv = 5,
                                n_iter = 50,
                                n_jobs = 10)
    best_model = model.fit(x_train, y_train)
    
    # Feature Importances plot for training data
    feat_importances = pd.Series(model.best_estimator_.feature_importances_, index=x_train.columns)
    feat_importances.nlargest(20).plot(kind='barh',color='r')
    plt.title("Feature Importances",fontsize=20)
    plt.xlabel("F Score")
    plt.show()
    
    dtree_params = model.best_params_
    y_train_pred = best_model.predict(x_train)
    print()
    print("Training WMAE:\t",mean_absolute_error(y_train,y_train_pred,train_weights))    
    return best_model, dtree_params

def random_Forest(X_train,Y_train,X_test,Y_test,train_weights,dtree_params):
    
    clf = RandomForestRegressor()
    
    param_grid = {
            'bootstrap': [True],
            'criterion': ['mse'],
            'max_depth': [dtree_params['max_depth']-5,
                          dtree_params['max_depth'],
                          dtree_params['max_depth']+5],
            'max_features': [dtree_params['max_features']],
            'max_leaf_nodes': [None],
            'min_impurity_decrease': [0.0],
            'min_impurity_split': [None],
            'min_samples_leaf': [4,5,6],
            'min_samples_split': [dtree_params['min_samples_split']],
            'min_weight_fraction_leaf': [0.0],
            'n_estimators': [10],
            'oob_score': [False],
            'warm_start': [False]
            }

    rfc_gscv = RandomizedSearchCV(clf, param_grid, cv=5,n_iter=8,n_jobs=10)
    print("Running grid search...")
    rfc_gscv.fit(X_train, Y_train)
    y_train_pred = rfc_gscv.predict(X = X_train)
    rf_params = rfc_gscv.best_params_
    print("Training WMAE:\t",mean_absolute_error(y_train,y_train_pred,train_weights))
    
    # Feature Importances plot for training data
    feat_importances = pd.Series(rfc_gscv.best_estimator_.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(20).plot(kind='barh',color='r')
    plt.title("Feature Importances",fontsize=20)
    plt.xlabel("F Score")
    plt.show()
    
    y_test_pred = rfc_gscv.predict(X = X_test)
    return y_test_pred,rf_params

# =============================================================================
# =============================================================================
# # Running code
# =============================================================================
# =============================================================================

print("Clean and Test train split...")
x_train, x_test, y_train, y_test, train_weights, test_weights = clean_data(raw_data)

# =============================================================================
# PCA
# =============================================================================

print("PCA...")
x_train_pca, x_test_pca = pca(x_train, x_test, 0.95)

# =============================================================================
# Decision tree regression
# =============================================================================

print("Decision Tree...")
fit_dt,dtree_params = decision_tree(normalize_data(x_train),y_train.iloc[:,0],train_weights.iloc[:,0])

file = open('dtree_parameters.txt','w') 
file.write(str(dtree_params))
file.close()

print("Test Data")
y_test_pred_dt = fit_dt.predict(normalize_data(x_test))
dec_tree_wmae = model_evaluation(y_test.iloc[:,0],y_test_pred_dt,test_weights,x_test)


feat_importances = pd.Series(fit_dt.best_estimator_.feature_importances_, index=x_train.columns)
feat_importances.nlargest(20).plot(kind='barh',color='r')
plt.title("Feature Importances",fontsize=20)
plt.xlabel("F Score")
plt.show()
# =============================================================================
# Random Forest
# =============================================================================

print("Random forest...")
y_test_pred_rf,rf_params = random_Forest(  normalize_data(x_train),
                                           y_train.iloc[:,0],
                                           normalize_data(x_test),
                                           y_test.iloc[:,0],
                                           train_weights.iloc[:,0],
                                           dtree_params)

file = open('rf_parameters.txt','w') 
file.write(str(rf_params))
file.close()

print(rf_params)
rf_wmae = model_evaluation(y_test.iloc[:,0],y_test_pred_rf,test_weights,x_test)

# =============================================================================
# XGBoost
# =============================================================================

print("Running XGBoost...")
init_model = XGBRegressor(
        n_jobs=-1,
        learning_rate =0.1,
        n_estimators=70,                # 70 optimal from hand tuning
        max_depth=40,                   # 40 optimal from hand tuning
        min_child_weight=1,
        gamma=10,                       # 10 optimal from hand tuning
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        importance_type = 'total_gain')

fit_xgb = Gradient_Boosted_fit(init_model,normalize_data(x_train),y_train.iloc[:,0],train_weights.iloc[:,0])
#fit_xgb.save_model('0.718Model.model')

print("Test Data")
y_test_pred_xgb = fit_xgb.predict(np.array(normalize_data(x_test)))
xgb_wmae = model_evaluation(y_test.iloc[:,0],y_test_pred_xgb,test_weights,x_test)

# =============================================================================
# Compare
# =============================================================================

# Feature Importances plot for training data
WMAE = pd.Series([dec_tree_wmae,rf_wmae,xgb_wmae], index=['Decision Tree',"Random Forest",'XGBoost'])
WMAE.plot(kind='bar',color='r')
plt.title("Weighted Mean Absolute Error by Model")
plt.ylabel("WMAE")
plt.show()
