#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:28:45 2019

@author: manuelagirotti

inspiration from:
https://www.simonwenkel.com/2018/09/08/revisiting-ml-datasets-yacht-hydrodynamics.html#exploring-the-dataset-and-preprocessing
"""

import pandas as pd
import os
import time
import numpy as np
import matplotlib.pyplot as plt
#import pydotplus

#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pathlib import Path
#from mpl_toolkits.mplot3d import axes3d
#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#from sklearn import tree


from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
#from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MaxAbsScaler
#from sklearn.model_selection import cross_val_score
#from sklearn.gaussian_process.kernels import RBF


#from sklearn.linear_model import ElasticNet 
#from sklearn.linear_model import SGDRegressor
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
#from sklearn.ensemble import BaggingRegressor
#import xgboost as xgb
#from sklearn.neural_network import MLPRegressor # use of pytorch for the neural network task?





# Importing data
cwd = Path(os.getcwd())
p = cwd / 'DATA'
data = pd.read_csv(p / "yacht_hydro.csv") 


# boxplots
#for col in data.columns:
#    plt.figure()
#    plt.boxplot(data[col])
#    plt.title(col)

# scaling the data
data_scaled = data.copy()
scaler = MaxAbsScaler()
data_scaled.loc[:,:] = scaler.fit_transform(data)
scaler_params = scaler.get_params()

# We will eventually need the physical (unscaled) data to display the results
extract_scaling_function = np.ones((1,data_scaled.shape[1]))
extract_scaling_function = scaler.inverse_transform(extract_scaling_function)


pd.set_option('display.max_columns', 7)
#print(data_scaled.iloc[:3,:])

## Shuffling data
#data = data.sample(frac=1,random_state=0).reset_index(drop=True)
#
#
## Separating inputs from outputs
#X_data = np.array(data.iloc[:,:6])
#y_data = np.array(data.iloc[:,-1])

# The dataset
datasets = {}
y = data_scaled['Residuary resistance'].values.reshape(-1,1)
X = data_scaled.copy()
X.drop(['Residuary resistance'], axis=1, inplace=True)
X = X.values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,shuffle=True)
comment = 'original dataset; scaled; 6 inputs, 1 output'

dataset_id = 'scaled_raw' 
datasets[dataset_id] = {'X_train': X_train, 'X_test' : X_test,
                        'y_train': y_train, 'y_test' : y_test,
                        'scaler' : scaler,
                        'scaler_array' : extract_scaling_function,
                        'comment' : comment,
                        'dataset' : dataset_id}


## Decision Tree Regressor
#def train_test_decision_tree_regression(X_train, X_test,
#                                        y_train, y_test,
#                                        dataset_id):
#    decision_tree_regression = DecisionTreeRegressor(random_state=42)
#    grid_parameters_decision_tree_regression = {'max_depth' : [None, 3,5,7,9,10,11]}
#    start_time = time.time()
#    grid_obj = GridSearchCV(decision_tree_regression,
#                            param_grid=grid_parameters_decision_tree_regression,
#                            n_jobs=-1,cv = 5,
#                            verbose=1)
#    grid_fit = grid_obj.fit(X_train, y_train)
#    training_time = time.time() - start_time
#    best_decision_tree_regression = grid_fit.best_estimator_
#    prediction = best_decision_tree_regression.predict(X_test)
#    
#    # Create DOT data
#    dot_data = tree.export_graphviz(best_decision_tree_regression, out_file=None)
#
#    # Draw graph
#    graph = pydotplus.graph_from_dot_data(dot_data)  
#
#    # Show graph
#    Image(graph.create_png())
#
#    r2 = metrics.r2_score(y_test, prediction)
#    mse = metrics.mean_squared_error(y_test, prediction)
#    mae = metrics.mean_absolute_error(y_true=y_test, y_pred=prediction)
#    
#    i=0
#    
#    # metrics for true values
#    # r2 remains unchanged, mse, mea will change and cannot be scaled
#    # because there is some physical meaning behind it
#    prediction_true_scale = prediction * datasets[dataset_id]['scaler_array'][:,-(i+1)]
#    
#    y_test_true_scale = y_test * datasets[dataset_id]['scaler_array'][:,-(i+1)]
#    mae_true_scale = metrics.mean_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    medae_true_scale = metrics.median_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    mse_true_scale = metrics.mean_squared_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    
#    return {'Regression type' : 'Decision Tree Regression',
#            'model' : grid_fit,
##            'Predictions' : prediction,
#            'R2' : r2,'MSE' : mse, 'MAE' : mae,
#            'MSE_true_scale' : mse_true_scale,
#            'RMSE_true_scale' : np.sqrt(mse_true_scale),
#            'MAE_true_scale' : mae_true_scale,
#            'MedAE_true_scale' : medae_true_scale,
#            'Training time' : training_time,
#            'dataset' : str(dataset_id) + str(-(i+1))}
#
#prova = train_test_decision_tree_regression(X_train, X_test,
#                                            y_train, y_test,
#                                            'scaled_raw' )


## Elastic Net
#def train_test_elastic_net(X_train, X_test,
#                           y_train, y_test, dataset_id):
#    elastic_net_regression = ElasticNet(random_state=42)
#    grid_parameters_elastic_net_regression = {'alpha': np.linspace(0.1,8,100), 
#                                              'l1_ratio' : np.linspace(0.,1.,100)}
#    start_time = time.time()
#    grid_obj = GridSearchCV(elastic_net_regression,
#                            param_grid=grid_parameters_elastic_net_regression,
#                            n_jobs=-1,cv = 5,
#                            verbose=1)
#    grid_fit = grid_obj.fit(X_train, y_train)
#    training_time = time.time() - start_time
#    
#    best_elastic_net_regression = grid_fit.best_estimator_
#    prediction = best_elastic_net_regression.predict(X_test)
#    
#    r2 = metrics.r2_score(y_test, prediction)
#    mse = metrics.mean_squared_error(y_test, prediction)
#    mae = metrics.mean_absolute_error(y_true=y_test, y_pred=prediction)
#    
#    i=0
#    
#    # metrics for true values
#    # r2 remains unchanged, mse, mea will change and cannot be scaled
#    # because there is some physical meaning behind it
#    prediction_true_scale = prediction * datasets[dataset_id]['scaler_array'][:,-(i+1)]
#    
#    y_test_true_scale = y_test * datasets[dataset_id]['scaler_array'][:,-(i+1)]
#    mae_true_scale = metrics.mean_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    medae_true_scale = metrics.median_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    mse_true_scale = metrics.mean_squared_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    
#    return {'Regression type' : 'Elastic Net Regression',
#            'model' : grid_fit,
#            'plot' : grid_fit.cv_results_.get('mean_test_score'),
#            'Predictions' : prediction.reshape(-1,1),
#            'R2' : r2,'MSE' : mse, 'MAE' : mae,
#            'MSE_true_scale' : mse_true_scale,
#            'RMSE_true_scale' : np.sqrt(mse_true_scale),
#            'MAE_true_scale' : mae_true_scale,
#            'MedAE_true_scale' : medae_true_scale,
#            'Training time' : training_time,
#            'dataset' : str(dataset_id)}
#
#
#enet_dict = train_test_elastic_net(X_train, X_test,
#                                        y_train, y_test,
#                                        'scaled_raw' )
#
#
##preds  = enet_dict['Predictions']
##
##xx = np.linspace(-.05,1.,100)
##
##plt.figure(figsize=(8,8))
##plt.plot(y_test, preds,'o')
##plt.plot(xx,xx,'-')
###plt.errorbar(L1ratio, error_list, yerr=error_std)
### plt.grid(True) # add a grid
##plt.xlabel('True Residuary Resistance')
##plt.ylabel('Predicted Residuary Resistance')
##plt.title('Real values vs prediction with Elastic Net Regressor')
##plt.axis('equal')
##plt.show()
#
#
#(xvar, yvar) = np.meshgrid(np.linspace(.1,8,100), np.linspace(0.,1.,100))
#
#Resplot = np.array(enet_dict['plot']).reshape(-1,1)
#zvar = Resplot.reshape(100,100)
##print(zvar)
#
#fig = plt.figure(num=1, clear=True,figsize=(8,8))
#ax = fig.add_subplot(1, 1, 1, projection='3d')
#
#ax.plot_surface(xvar, yvar, zvar,cmap=cm.plasma)
#ax.set(xlabel='alpha', ylabel='l1 ratio', title='Mean_test_score')
#ax.view_init(azim=30, elev=30) 
#
#fig.tight_layout()

#plt.show()

#plt.figure()
#plt.plot(L1ratio, error_list)
##plt.errorbar(L1ratio, error_list, yerr=error_std)
## plt.grid(True) # add a grid
#plt.xlabel('L1 to L2 ratio')
#plt.title('R^2 error for different values of L1_ratio')
#plt.show()


## k-NN
#def train_test_kNN(X_train, X_test, y_train, y_test, dataset_id):
#    kNN_regression = KNeighborsRegressor(p=2, metric='minkowski')
#    grid_parameters_kNN_regression = {'n_neighbors': np.arange(1,51), 
#                                      'weights': ['uniform', 'distance']}
#    start_time = time.time()
#    grid_obj = GridSearchCV(kNN_regression,
#                            param_grid=grid_parameters_kNN_regression,
#                            n_jobs=-1,cv = 5,
#                            verbose=1)
#    grid_fit = grid_obj.fit(X_train, y_train)
#    training_time = time.time() - start_time
#    
#    best_kNN_regression = grid_fit.best_estimator_
#    print(best_kNN_regression)
#    prediction = best_kNN_regression.predict(X_test)
#    
#    r2 = metrics.r2_score(y_test, prediction)
#    mse = metrics.mean_squared_error(y_test, prediction)
#    mae = metrics.mean_absolute_error(y_true=y_test, y_pred=prediction)
#    
#    i=0
#    
#    # metrics for true values
#    # r2 remains unchanged, mse, mea will change and cannot be scaled
#    # because there is some physical meaning behind it
#    prediction_true_scale = prediction * datasets[dataset_id]['scaler_array'][:,-(i+1)]
#    
#    y_test_true_scale = y_test * datasets[dataset_id]['scaler_array'][:,-(i+1)]
#    mae_true_scale = metrics.mean_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    medae_true_scale = metrics.median_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    mse_true_scale = metrics.mean_squared_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    
#    return {'Regression type' : 'k-NN Regression',
#            'model' : grid_fit,
#            'plot' : grid_fit.cv_results_.get('mean_test_score'),
#            'Predictions' : prediction.reshape(-1,1),
#            'R2' : r2,'MSE' : mse, 'MAE' : mae,
#            'MSE_true_scale' : mse_true_scale,
#            'RMSE_true_scale' : np.sqrt(mse_true_scale),
#            'MAE_true_scale' : mae_true_scale,
#            'MedAE_true_scale' : medae_true_scale,
#            'Training time' : training_time,
#            'dataset' : str(dataset_id)}
#
#    
#kNN_dict = train_test_kNN(X_train, X_test, y_train, y_test,'scaled_raw')
#
#
##GSplot = np.array(kNN_dict['plot']).reshape(-1,1)
##xx = np.linspace(1,51)
##uni_w = GSplot[::2]
##dist_w = GSplot[1::2]
##
##fig = plt.figure(num=1, clear=True,figsize=(8,8))
##plt.plot(xx,uni_w)
##plt.plot(xx,dist_w)
##plt.xlabel('k nearest neighbours')
##plt.title('mean_test_score')
##plt.legend(('Uniform','Distance'))
##plt.show()
#
#
#preds  = kNN_dict['Predictions']
#
#xx = np.linspace(-.05,1.,100)
#
#plt.figure(figsize=(8,8))
#plt.plot(y_test, preds,'o')
#plt.plot(xx,xx,'-')
##plt.errorbar(L1ratio, error_list, yerr=error_std)
## plt.grid(True) # add a grid
#plt.xlabel('True Residuary Resistance')
#plt.ylabel('Predicted Residuary Resistance')
#plt.title('Real values vs prediction with kNN Regressor')
#plt.axis('equal')
#plt.show()


## MLP
#def train_test_MLP(X_train, X_test, y_train, y_test, dataset_id):
#    MLP_regression = MLPRegressor(solver = 'lbfgs', random_state=42)
#    grid_parameters_MLP_regression = {'hidden_layer_sizes':[(9,5),(30,20),(100,50)], 
#                                    'activation':['logistic', 'relu','tanh'], 
##                                    'learning_rate':['constant', 'adaptive'], #adaptive only for sgd
#                                    'alpha': np.linspace(0,1.0,10), #regularizer
##                                    'early_stopping':[False,True]#early stopping only for sgd
#                                    } 
#    start_time = time.time()
#    grid_obj = GridSearchCV(MLP_regression,
#                            param_grid=grid_parameters_MLP_regression,
#                            n_jobs=-1,cv = 5,
#                            verbose=1)
#    grid_fit = grid_obj.fit(X_train, y_train)
#    training_time = time.time() - start_time
#    
#    best_MLP_regression = grid_fit.best_estimator_
#    print(best_MLP_regression)
#    prediction = best_MLP_regression.predict(X_test)
#    
#    r2 = metrics.r2_score(y_test, prediction)
#    mse = metrics.mean_squared_error(y_test, prediction)
#    mae = metrics.mean_absolute_error(y_true=y_test, y_pred=prediction)
#    
#    i=0
#    
#    # metrics for true values
#    # r2 remains unchanged, mse, mea will change and cannot be scaled
#    # because there is some physical meaning behind it
#    prediction_true_scale = prediction * datasets[dataset_id]['scaler_array'][:,-(i+1)]
#    
#    y_test_true_scale = y_test * datasets[dataset_id]['scaler_array'][:,-(i+1)]
#    mae_true_scale = metrics.mean_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    medae_true_scale = metrics.median_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    mse_true_scale = metrics.mean_squared_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    
#    return {'Regression type' : 'MLP Regression',
#            'model' : grid_fit,
##            'plot' : grid_fit.cv_results_.get('mean_test_score'),
#            'Predictions' : prediction.reshape(-1,1),
#            'R2' : r2,'MSE' : mse, 'MAE' : mae,
#            'MSE_true_scale' : mse_true_scale,
#            'RMSE_true_scale' : np.sqrt(mse_true_scale),
#            'MAE_true_scale' : mae_true_scale,
#            'MedAE_true_scale' : medae_true_scale,
#            'Training time' : training_time,
#            'dataset' : str(dataset_id)}
#
#    
#MLP_dict = train_test_MLP(X_train, X_test, np.array(y_train), np.array(y_test),'scaled_raw')
#
#
##GSplot = np.array(kNN_dict['plot']).reshape(-1,1)
##xx = np.linspace(1,51)
##uni_w = GSplot[::2]
##dist_w = GSplot[1::2]
##
##fig = plt.figure(num=1, clear=True,figsize=(8,8))
##plt.plot(xx,uni_w)
##plt.plot(xx,dist_w)
##plt.xlabel('k nearest neighbours')
##plt.title('mean_test_score')
##plt.legend(('Uniform','Distance'))
##plt.show()
#
#
#preds  = MLP_dict['Predictions']
#
#xx = np.linspace(-.05,1.,100)
#
#plt.figure(figsize=(8,8))
#plt.plot(y_test, preds,'o')
#plt.plot(xx,xx,'-')
##plt.errorbar(L1ratio, error_list, yerr=error_std)
## plt.grid(True) # add a grid
#plt.xlabel('True Residuary Resistance')
#plt.ylabel('Predicted Residuary Resistance')
#plt.title('Real values vs prediction with fully-connected NN Regressor')
#plt.axis('equal')
#plt.show()


## Random Forest
#def train_test_RForest(X_train, X_test, y_train, y_test, dataset_id):
#    RForest_regression = RandomForestRegressor(bootstrap=True, random_state=42)
#    grid_parameters_RForest_regression = {'n_estimators':[10,20,40,60,80,100,
#                                                          120,140,160,180,200,
#                                                          220,240,260,280,300,
#                                                          320,340,360,380,400,
#                                                          420,440,460,480,500], 
#                                    'max_depth':[3,5,10,50,100]
#                                    } 
#    start_time = time.time()
#    grid_obj = GridSearchCV(RForest_regression,
#                            param_grid=grid_parameters_RForest_regression,
#                            n_jobs=-1,cv = 5,
#                            verbose=1)
#    grid_fit = grid_obj.fit(X_train, np.ravel(y_train))
#    training_time = time.time() - start_time
#    
#    best_RForest_regression = grid_fit.best_estimator_
#    print(best_RForest_regression)
#    prediction = best_RForest_regression.predict(X_test)
#    
#    r2 = metrics.r2_score(y_test, prediction)
#    mse = metrics.mean_squared_error(y_test, prediction)
#    mae = metrics.mean_absolute_error(y_true=y_test, y_pred=prediction)
#    
#    i=0
#    
#    # metrics for true values
#    # r2 remains unchanged, mse, mea will change and cannot be scaled
#    # because there is some physical meaning behind it
#    prediction_true_scale = prediction * datasets[dataset_id]['scaler_array'][:,-(i+1)]
#    
#    y_test_true_scale = y_test * datasets[dataset_id]['scaler_array'][:,-(i+1)]
#    mae_true_scale = metrics.mean_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    medae_true_scale = metrics.median_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    mse_true_scale = metrics.mean_squared_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
#    
#    return {'Regression type' : 'Random Forest Regression',
#            'model' : grid_fit,
#            'plot' : grid_fit.cv_results_.get('mean_test_score'),
#            'Predictions' : prediction.reshape(-1,1),
#            'R2' : r2,'MSE' : mse, 'MAE' : mae,
#            'MSE_true_scale' : mse_true_scale,
#            'RMSE_true_scale' : np.sqrt(mse_true_scale),
#            'MAE_true_scale' : mae_true_scale,
#            'MedAE_true_scale' : medae_true_scale,
#            'Training time' : training_time,
#            'dataset' : str(dataset_id)}
#
#    
#RForest_dict = train_test_RForest(X_train, X_test, y_train, y_test,'scaled_raw')
#
#
#RFplot = np.array(RForest_dict['plot']).reshape(-1,1)
#xx = [10,20,40,60,80,100,120,140,160,180,200,220,240,260,
#            280,300,320,340,360,380,400,420,440,460,480,500]
##np.arange(1,27)
#RF_depth3 = RFplot[:26]
#RF_depth5 = RFplot[26:52]
#RF_depth10 = RFplot[52:78]
#RF_depth50 = RFplot[78:104]
#RF_depth100 = RFplot[104:]
#
#fig = plt.figure(num=1, clear=True,figsize=(8,8))
#plt.plot(xx,RF_depth3)
#plt.plot(xx,RF_depth5)
#plt.plot(xx,RF_depth10)
#plt.plot(xx,RF_depth50)
#plt.plot(xx,RF_depth100)
##plt.xticks([10,20,40,60,80,100,120,140,160,180,200,220,240,260,
##            280,300,320,340,360,380,400,420,440,460,480,500])
#plt.xlabel('Number of trees')
#plt.title('mean_test_score')
#plt.legend(('Depth=3','Depth=5', 'Depth=10', 'Depth=50', 'Depth=100'))
#plt.show()


#preds  = RForest_dict['Predictions']
#
#xx = np.linspace(-.05,1.,100)
#
#plt.figure(figsize=(8,8))
#plt.plot(y_test, preds,'o')
#plt.plot(xx,xx,'-')
##plt.errorbar(L1ratio, error_list, yerr=error_std)
## plt.grid(True) # add a grid
#plt.xlabel('True Residuary Resistance')
#plt.ylabel('Predicted Residuary Resistance')
#plt.title('Real values vs prediction with Random Forest Regressor')
#plt.axis('equal')
#plt.show()


# AdaBoost
def train_test_AdaBoost(X_train, X_test, y_train, y_test, dataset_id):
    AdaB_regression = AdaBoostRegressor(random_state=42)
    grid_parameters_AdaB_regression = {'n_estimators':[10,20,30,40,50,60,70,80,90,100,
                                                       110,120,130,140,150,160,170,180,190,200],
#                                    'learning_rate':np.linspace(0.1,1.5,10),
                                    'loss' : ['linear', 'square', 'exponential']
                                    } 
    start_time = time.time()
    grid_obj = GridSearchCV(AdaB_regression,
                            param_grid=grid_parameters_AdaB_regression,
                            n_jobs=-1,cv = 5,
                            verbose=1)
    grid_fit = grid_obj.fit(X_train, np.ravel(y_train))
    training_time = time.time() - start_time
    
    best_AdaB_regression = grid_fit.best_estimator_
    print(best_AdaB_regression)
    prediction = best_AdaB_regression.predict(X_test)
    
    r2 = metrics.r2_score(y_test, prediction)
    mse = metrics.mean_squared_error(y_test, prediction)
    mae = metrics.mean_absolute_error(y_true=y_test, y_pred=prediction)
    
    i=0
    
    # metrics for true values
    # r2 remains unchanged, mse, mea will change and cannot be scaled
    # because there is some physical meaning behind it
    prediction_true_scale = prediction * datasets[dataset_id]['scaler_array'][:,-(i+1)]
    
    y_test_true_scale = y_test * datasets[dataset_id]['scaler_array'][:,-(i+1)]
    mae_true_scale = metrics.mean_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
    medae_true_scale = metrics.median_absolute_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
    mse_true_scale = metrics.mean_squared_error(y_true=y_test_true_scale, y_pred=prediction_true_scale)
    
    return {'Regression type' : 'AdaBoost Regression',
            'model' : grid_fit,
            'plot' : grid_fit.cv_results_.get('mean_test_score'),
            'Predictions' : prediction.reshape(-1,1),
            'R2' : r2,'MSE' : mse, 'MAE' : mae,
            'MSE_true_scale' : mse_true_scale,
            'RMSE_true_scale' : np.sqrt(mse_true_scale),
            'MAE_true_scale' : mae_true_scale,
            'MedAE_true_scale' : medae_true_scale,
            'Training time' : training_time,
            'dataset' : str(dataset_id)}

    
AdaB_dict = train_test_AdaBoost(X_train, X_test, y_train, y_test,'scaled_raw')


ABplot = np.array(AdaB_dict['plot']).reshape(-1,1)
xx = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
AB_lin = ABplot[:20]
AB_sq = ABplot[20:40]
AB_exp = ABplot[40:]


fig = plt.figure(num=1, clear=True,figsize=(8,8))
plt.plot(xx,AB_lin)
plt.plot(xx,AB_sq)
plt.plot(xx,AB_exp)
plt.xlabel('Number of trees')
plt.title('mean_test_score')
plt.legend(('Linear','Square', 'Exponential'))
plt.show()


#preds  = AdaB_dict['Predictions']
#
#xx = np.linspace(-.05,1.,100)
#
#plt.figure(figsize=(8,8))
#plt.plot(y_test, preds,'o')
#plt.plot(xx,xx,'-')
##plt.errorbar(L1ratio, error_list, yerr=error_std)
## plt.grid(True) # add a grid
#plt.xlabel('True Residuary Resistance')
#plt.ylabel('Predicted Residuary Resistance')
#plt.title('Real values vs prediction with Random Forest Regressor')
#plt.axis('equal')
#plt.show()

