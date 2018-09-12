
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import PLSCanonical
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import time
import json
import sys
import argparse
from argparse import ArgumentParser
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import copy
import xlrd


# In[ ]:


def classifier(X, Y, subj, num_samples, mean,deviation, ddof, method, feature_selection, model_params,
               num_features, step, verbose_train, scaling):
    #process the labels
    if scaling:
        Y_temp = np.array(Y)
        Y_temp = descale_data(matrix=Y_temp, deviation=deviation, ddof=ddof) #descale
        Y_temp = Y_temp+mean                             #add mean
        for j in range(len(Y_temp)):
            if Y_temp[j]>1:
                Y_temp[j]=1 
            elif Y_temp[j]<0:
                Y_temp[j]=0  
    else:
        Y_temp = Y+mean
    Y_temp=round_num(Y_temp)

    #choose, which classifier to use
    if method== 'rf':#build a generic Random Forest
        model = RandomForestRegressor(n_estimators=model_params['n_trees'], random_state=0, 
                                      max_features=model_params['max_features'], max_depth=model_params['max_depth'], 
                                      min_samples_leaf=model_params['min_samples_leaf'])
    elif method == 'pls':
        model = PLSRegression(n_components=model_params['n_comp'], scale=False) #initialize a generic PLS model
        
    elif method == 'svm':
        model = SVR(C= model_params['C'], epsilon=model_params['epsilon'], kernel=model_params['kernel'], 
                   gamma = model_params['gamma'], degree=model_params['degree'])
    elif method == 'lda':
        model = LinearDiscriminantAnalysis()
    if method == 'nn':
        from keras.wrappers.scikit_learn import KerasRegressor
        from keras.models import Sequential
        from keras.layers import Dense, Activation
        import tensorflow
        coeff = np.zeros(X.shape[1])
        features = [True for f in range(X.shape[1])]
        estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
        model =estimator
        sfold = StratifiedKFold(n_splits=15)
        results = cross_val_score(estimator, X, Y_temp, cv=sfold)
        estimator.fit(X, Y_temp)
        print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
        auc = 0
        train_error = 0
    else:
        #choose a method for feature selection
        if feature_selection == 'FromModel':
            model_temp = copy.copy(model)
            if method=='svm' and model_params['kernel'] != 'linear':
                model_temp = SVR(C= 1, epsilon=1, kernel='linear')
            model_temp.fit(X, Y)
            model_temp = SelectFromModel(model, prefit=True)
            features = model_temp.get_support()
        elif feature_selection == 'rfecv':
            rfe = RFECV(estimator=model, cv=20, step=step)
            if method=='svm' and model_params['kernel'] != 'linear':
                rfe = SVR(C= 1, epsilon=1, kernel='linear')
            fit = rfe.fit(X, Y)
            features = fit.support_
        elif feature_selection == 'rfe':
            model_temp = copy.copy(model)
            if method=='svm' and model_params['kernel'] != 'linear':
                model_temp = SVR(C= 1, epsilon=1, kernel='linear')
            model_temp.fit(X, Y)
            rfe = RFE(estimator=model, n_features_to_select=num_features, step=step)
            fit = rfe.fit(X, Y)
            features = fit.support_
        elif feature_selection == 'pca':      
            pca = PCA(n_components = model_params['n_comp'])
            fit = pca.fit(X)
            features = (-np.mean(fit.components_, axis=0)).argsort()[:num_features]
        elif feature_selection == 'lda':
            clf = LinearDiscriminantAnalysis()
            clf.fit(X, Y_temp)
            features = clf.coef_.argsort()[:num_features]
        else:# feature_selection is None:
            features = [True for f in range(X.shape[1])]

        #transform input to suit new dimensions
        X_new = X[:, features]
        if len(X_new.shape) ==3:
            X_new =  X_new.reshape(X_new.shape[0], X_new.shape[2])

        #fit the model  
        if method=='lda':
            model.fit(X_new, Y_temp)
        elif method:
            model.fit(X_new, Y) #ACTUALLY getting the classifier model, fit model to data
            pred = model.predict(X_new) #predict the values on the train set

        #get coefficients
        if method == 'rf':
              coeff = model.feature_importances_.flatten()
        elif method == 'pls':
              coeff = model.coef_
              #print "What about ", model.x_scores_
              #print "Mean of coeff: ", np.mean(model.coef_)
              #print "Standard deviation of coeff: ", np.std(model.coef_)
              #print "Max of coeff: ", np.amax(model.coef_)
              #print "Min of coeff: ", np.amin(model.coef_)
              #print "Mean of new coeff: ", np.mean(coeff)
              #print "Standard deviation of coeff: ", np.std(coeff)
              #print "Max of new coeff: ", np.amax(coeff)
              #print "Min of new coeff: ", np.amin(coeff)
        elif method == 'svm': #and model_params['kernel']=='linear') :
              coeff = model.coef_.flatten()
        else:
              coeff = np.zeros(X.shape[1])

        #process the predictions
        pred_temp = pred.flatten()
        if feature_selection != 'lda' and scaling:
            pred_temp = descale_data(matrix=pred_temp, deviation=deviation, ddof=ddof ) #descale
            pred_temp = pred_temp+mean #add mean
        elif feature_selection != 'lda':
            pred_temp = pred_temp+mean
        pred_temp = round_num(pred_temp)

        train_error = mean_squared_error(y_true=Y_temp, y_pred=pred_temp)
        fpr, tpr, auc = get_roc_auc(labels=Y_temp, predictions=pred_temp)#AUC if needed

        if verbose_train:
            print "Training error: ", train_error/num_samples
            print "Training error computed in library: ", train_error_temp
            print "Training auc: ", auc
            plot_roc_curve(fpr, tpr, auc)
    
    return features, model, train_error, auc, coeff


# In[ ]:


def simple_classifier(X, Y, subj, num_samples, mean,deviation, ddof, method, feature_selection, model_params,
               num_features, step, verbose_train, scaling):
    if scaling:
        Y_temp = np.array(Y)
        Y_temp = descale_data(matrix=Y_temp, deviation=deviation, ddof=ddof) #descale
        Y_temp = Y_temp+mean                             #add mean
        for j in range(len(Y_temp)):
            if Y_temp[j]>1:
                Y_temp[j]=1 
            elif Y_temp[j]<0:
                Y_temp[j]=0
                Y_temp=Y_temp.astype(int)
    else:
        Y_temp = Y+mean
    model = PLSRegression(n_components=model_params['n_comp'], scale=False) #initialize a generic PLS model
    model_temp = copy.copy(model)
    model_temp.fit(X, Y)
    rfe = RFE(estimator=model_temp, n_features_to_select=num_features, step=step)
    fit = rfe.fit(X, Y)
    features = fit.support_
    X_new = X[:, features]
    model.fit(X_new, Y) #ACTUALLY getting the classifier model, fit model to data
    pred = model.predict(X_new) #predict the values on the train set
    pred_temp = pred.flatten()
    if scaling:
        pred_temp = descale_data(matrix=pred_temp, deviation=deviation, ddof=ddof ) #descale
        pred_temp = pred_temp+mean #add mean
        pred_temp = round_num(pred_temp)
    pred_temp = pred_temp+mean
    train_error = mean_squared_error(y_true=Y_temp, y_pred=pred_temp)
    fpr, tpr, auc = get_roc_auc(labels=Y_temp, predictions=pred_temp)#AUC if needed
    return features, model, train_error, auc, coeff


# In[ ]:

def round_num(numbers):
	res = copy.copy(numbers)
	for ix in range(len(numbers)):
		if numbers[ix] >= 1:
			res[ix] = 1
		elif numbers[ix] <=0:
			res[ix] = 0
		else:
			res[ix] = np.rint(numbers[ix])
	return res


def center_data(matrix, mean=None):
    if mean is None:
        matrix = matrix - np.mean(a=matrix, axis=0)
    else:
        matrix = matrix - mean
    return matrix


# In[ ]:


def scale_data(matrix, deviation, ddof):
    if deviation is None:
        deviation= np.std(a=matrix, axis=0, ddof=ddof)
    matrix_temp = copy.copy(matrix)
    if np.isnan(matrix).any():
        matrix = matrix_temp
    elif np.count_nonzero(deviation)==0:
        matrix = matrix_temp
    else:
        matrix = matrix/deviation
    return matrix


# In[ ]:


def descale_data(matrix, deviation, ddof):
    if deviation is None:#use the deviation from the same matrix to scale
        matrix = matrix*np.std(a=matrix, axis=0, ddof=ddof)
    else:#use deviation given in parameters
        matrix = matrix*deviation
    return matrix


# In[ ]:


def get_roc_auc(labels, predictions):
    #compute precision-recall curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true=labels, y_score=predictions, drop_intermediate=False) 
    #compute area under the curve for this run
    auc = metrics.roc_auc_score(y_true=labels, y_score=predictions, average='macro', sample_weight=None)
    return fpr, tpr, auc


# In[ ]:


def plot_roc_curve(fpr, tpr, auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[ ]:


#MULTILEVEL PLS
def perform_multilevel_pls(X, Y, subj, unique_subj, num_unique_subj, num_subj, scaling, par, ddof, method, 
                           feature_selection, num_features, step, model_params, verbose_train):
    
    Xb= np.zeros(X.shape) 
    Xw= np.zeros(X.shape)
    #mean-centering
    X_centered = center_data(matrix=X, mean=None) #mean centering of data
    Y_centered = center_data(matrix=Y, mean=None) 

    #split matrix into between and within subject variations
    if par!='all':
        Xb, Xw = split_between_within_subject_matrix(X=X_centered, subj=subj, unique_subj=unique_subj, 
                                                 num_unique_subj=num_unique_subj, num_subj=num_subj)
    
    #scaling (if necessary and specified)
    if scaling: #scale the data, if the flag is true
        X_scaled = scale_data(matrix=X_centered, deviation=None, ddof=ddof)
        Y_scaled = scale_data(matrix=Y_centered, deviation=None, ddof=ddof)                      
        if par!='all':
            Xb_scaled = scale_data(matrix=Xb, deviation=None, ddof=ddof)
            Xw_scaled = scale_data(matrix=Xw, deviation=None, ddof=ddof)
    else:
        X_scaled = X_centered
        Y_scaled = Y_centered
        if par!='all':
            Xb_scaled = Xb
            Xw_scaled = Xw
    
    #which matrix are we interested in
    if par=='all':
        X_target = X_scaled
    elif par=='between':
        X_target = Xb_scaled
    elif par=='within':
        X_target = Xw_scaled
    
    #perform classifier (PLS or other) on target data
    features, model, train_error, train_auc, coeff= classifier(X=X_target, Y=Y_scaled, subj=subj, num_samples=num_subj,
                                                           mean=np.mean(Y), deviation= np.std(Y, axis=0, ddof=ddof), 
                                                           ddof=ddof, method=method, 
                                                           feature_selection=feature_selection, 
                                                           num_features=num_features, step=step, 
                                                           model_params=model_params, verbose_train=verbose_train, 
                                                           scaling=scaling)
    
    return features, model, train_error, train_auc, Xb, Xw, Y_scaled, X_scaled, coeff


# In[ ]:


def separate_train_test(X, Y, subj, train_subj, test_subj, num_subj):
    X_train = None #initialization
    X_test = None
    Y_train = []
    Y_test = []
    subj_train = []
    subj_test = []  
    
    #create the test and train dataset from matrix X with chosen subjects
    mask = np.isin(subj,train_subj)
    inverse_mask = np.invert(mask)

    subj_train = subj[mask]
    subj_test = subj[inverse_mask]
    X_train = X[mask]
    Y_train = Y[mask]
    X_test = X[inverse_mask]
    Y_test = Y[inverse_mask]

    num_train_subj = len(subj_train)          #how many entries in train dataset
    num_test_subj= num_subj - num_train_subj  #how many entries in test dataset
        
    return X_train, X_test, Y_train, Y_test, subj_train, subj_test, num_train_subj, num_test_subj


# In[ ]:


def full_script(num_folds, num_repeats, scaling, num_permutations, par, filename, verbose, ddof, method, 
                feature_selection,num_features,step, model_params,verbose_train, mode):
    
    #READ OR MAKE UP DATA
    X, Y, subj, IDs, NetCalc, frQ = read_data(filename, mode)
    
    #create list of unique subjects
    unique_subj = np.unique(subj) 
    #the number of entries = number of subjects corresponding to entries (1 subject per entry)
    num_subj = len(subj)    
    #the number of unique subjects
    num_unique_subj = len(unique_subj) 
    #initialization of errors
    error = 0 
    permutation_error=None
    permutation_auc=None 
    permutation_Q=None
    full_train_error=0
    full_train_auc=0
    crossval_error=0
    crossval_auc=0
    crossval_Q=0
    crossval_train_err=0
    crossval_train_auc=0
    

    #PERFORMING MULTILEVEL PLS ON WHOLE DATASET

    features, full_model, full_train_error, full_train_auc, Xb, Xw, Y_scaled, X_scaled, coeff= perform_multilevel_pls(X=X, Y=Y, subj=subj, 
                           unique_subj=unique_subj, num_subj=num_subj, 
                           num_unique_subj=num_unique_subj, scaling=scaling, par=par, 
                           ddof=ddof, method=method, feature_selection=feature_selection, 
                           num_features=num_features, step=step, model_params=model_params, verbose_train=verbose_train)

   
    if method=='pls':
        x =  np.matmul(X_scaled[:,features],full_model.x_weights_[:, 0])
        y =  np.matmul(X_scaled[:,features],full_model.x_weights_[:, 1])
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.xlabel('PLS component 1')
        plt.ylabel('PLS component 2')
        colors = ['green', 'orange']
        ax.scatter(x,y,c=Y)

        fig.savefig(mode+"_"+str(model_params['n_comp'])+"_"+str(feature_selection)+'.png')   # save the figure to file
        plt.close(fig)   
        
    #print "Features chosen: ", IDs[features]
    #print "Biomarkers chosen: ", NetCalc[features]
    #print "rfQ chosen: ", frQ[features]
    #print "Coefficients: ", coeff
    
    #print "!Mean of new coeff: ", np.mean(coeff)
    #print "!Standard deviation of coeff: ", np.std(coeff)
    #print "!Max of new coeff: ", np.amax(coeff)
    #print "!Min of new coeff: ", np.amin(coeff)
    IDs = np.array(IDs)
    coefficients = {}
    for id in IDs:
        coefficients[id]=0
    for i in range(len(IDs[features])):
        coefficients[IDs[i]]=coeff[i]
    #print "###!Mean of new coeff: ", np.mean(coefficients.values())
    #print "###!Standard deviation of coeff: ", np.std(coefficients.values())
    #print "###!Max of new coeff: ", np.amax(coefficients.values())
    #print "###!Min of new coeff: ", np.amin(coefficients.values())
    if verbose: 
        print "CROSS_VALIDATION ON ACTUAL DATA: "
        
    #CROSS-VALIDATION
    crossval_error, crossval_auc, crossval_Q, crossval_R, crossval_acc, crossval_F, crossval_train_err, crossval_train_auc =    cross_validation(X=X, Y=Y, subj=subj, unique_subj=unique_subj, num_subj=num_subj, num_unique_subj=num_unique_subj, 
                     num_folds=num_folds, num_repeats=num_repeats, scaling=scaling, par=par, 
                     verbose=verbose, ddof=ddof, method=method, feature_selection=feature_selection, 
                     num_features = num_features, step=step, model_params=model_params, verbose_train=verbose_train)
  
    if verbose: 
        print "PERMUTATED DATA CROSS_VALIDATION: "
        
    #PERMUTATE
    permutation_error, permutation_auc, permutation_Q, permutation_train_error, permutation_train_auc =                                                        validate_permutation(X=X, Y=Y, subj=subj,
                                                                             unique_subj=unique_subj, 
                                                                             num_subj=num_subj,
                                                                             num_unique_subj=num_unique_subj,
                                                                             num_folds=num_folds,
                                                                             num_repeats=num_repeats, 
                                                                             scaling=scaling,
                                                                             num_permutations=num_permutations, 
                                                                             par=par, verbose=verbose, 
                                                                             ddof=ddof,method=method,
                                                                             feature_selection=feature_selection, 
                                                                             num_features=num_features, step=step,
                                                                             model_params=model_params, 
                                                                             verbose_train=verbose_train)
   
    results = {'num_folds':num_folds,'num_repeats':num_repeats,'scaling':scaling,'num_permutations':num_permutations,
               'par':par, 'filename':filename, 'verbose':verbose, 'ddof':ddof, 'method':method, 
               'feature_selection':feature_selection, 'num_features':num_features, 'step':step, 
               'n_trees':model_params['n_trees'], 'max_depth':model_params['max_depth'], 
               'max_features':model_params['max_features'], 'min_samples_leaf':model_params['min_samples_leaf'],
               'num_comp':model_params['n_comp'], 'full_train_error':full_train_error, 'full_train_auc':full_train_auc, 
               'crossval_error':crossval_error, 'crossval_auc':crossval_auc, 'crossval_Q':crossval_Q, 'crossval_R':crossval_R,
                'crossval_acc':crossval_acc, 'crossval_F':crossval_F, 
               'crossval_train_err': crossval_train_err, 'crossval_train_auc':crossval_train_auc, 
               'permutation_error': permutation_error, 'permutation_auc':permutation_auc, 'permutation_Q':permutation_Q, 
               'C': model_params['C'], 'gamma': model_params['gamma'], 'epsilon': model_params['epsilon'], 
               'degree': model_params['degree'], 'kernel': model_params['kernel']}
    
    return results, coefficients.values(), IDs


# In[ ]:


#CROSS-VALIDATION
def cross_validation(X, Y, subj, unique_subj, num_subj, num_unique_subj, num_folds, num_repeats, scaling, par, 
                     verbose, ddof, method, feature_selection, num_features, step, model_params,
                     verbose_train):
    #initialization
    error       = 0 
    auc         = 0
    train_error = 0
    train_auc   = 0
    
    #create array of labels for each subject
    Y_temp = [Y[np.where(subj==unique_subj[i])[0][0]] for i in range(num_unique_subj)]
    Y=np.array(Y)
    new_Y = np.zeros(Y.shape)
    weird_Y = np.zeros(Y.shape)
    #compare_Y = np.zeros(Y.shape)
    #repeat cross_validation as many times as specified
    for i in range(num_repeats): 
        kf = StratifiedKFold(n_splits=num_folds)
        
        for train_index, test_index in kf.split(X[:num_unique_subj], Y_temp):#repeat with every fold as test set
            train_subj = unique_subj[train_index]
            test_subj = unique_subj[test_index]
        
            X_train, X_test, Y_train, Y_test, subj_train, subj_test, num_train_subj, num_test_subj = separate_train_test(X=X, Y=Y, subj=subj, train_subj=train_subj,
                                                                        test_subj=test_subj, num_subj=num_subj)

            num_unique_train_subj=len(train_subj)
            num_unique_test_subj=len(test_subj)
            
            X_train_mean = np.mean(X_train, axis=0)
            Y_train_mean = np.mean(Y_train, axis=0)

            features, model, train_error_temp, train_auc_temp, Xb_train, Xw_train, Y_scaled_train, X_scaled_train, coeff = perform_multilevel_pls(X=X_train, Y=Y_train, subj=subj_train, 
                                                              unique_subj=train_subj, 
                                                              num_unique_subj=num_unique_train_subj, 
                                                              num_subj=num_train_subj, scaling=scaling, par=par, 
                                                              ddof=ddof, method=method, 
                                                              feature_selection=feature_selection, 
                                                              num_features=num_features, step=step,
                                                              model_params=model_params, 
                                                              verbose_train=verbose_train)

            X_centered_test = center_data(X_test, X_train_mean) #mean centering of data
            Y_centered_test = center_data(Y_test, Y_train_mean)   

            #split test data into between and within subject variation
            Xb_test, Xw_test = split_between_within_subject_matrix(X=X_centered_test, subj=subj_test, 
                                                                   unique_subj=test_subj,
                                                                   num_unique_subj=num_unique_test_subj, 
                                                                   num_subj=num_test_subj )

                
            if scaling: #scale the data, if the flag is true
                X_train_deviation = np.std(X_train, axis = 0, ddof=ddof)
                Y_train_deviation = np.std(Y_train, axis = 0, ddof=ddof)
                Xb_train_deviation = np.std(Xb_train, axis = 0, ddof=ddof)
                Xw_train_deviation = np.std(Xw_train, axis = 0, ddof=ddof)

                X_scaled_test = scale_data(matrix=X_centered_test, deviation=X_train_deviation, ddof=ddof)
                Y_scaled_test = scale_data(matrix=Y_centered_test, deviation=Y_train_deviation, ddof=ddof)

                Xb_scaled_test = scale_data(matrix=Xb_test, deviation=Xb_train_deviation, ddof=ddof)
                Xw_scaled_test = scale_data(matrix=Xw_test, deviation=Xw_train_deviation, ddof=ddof)
            else:

                X_scaled_test = X_centered_test
                Y_scaled_test = Y_centered_test
                Xb_scaled_test = Xb_test
                Xw_scaled_test = Xw_test

            #if only between or only within subject variations chosen, predict on that part of the matrix
            if par=='all':
                X_target = X_scaled_test
            elif par=='between':
                X_target = Xb_scaled_test
            elif par=='within':
                X_target = Xw_scaled_test
            X_target = X_target[:, features]

            if len(X_target.shape)==3:   
                X_target = X_target.reshape(X_target.shape[0], X_target.shape[2]) 

            pred = model.predict(X_target) #predict test data with model trained on the training set

            #for i in range(len(test_index)):
            #    weird_Y[test_index[i]] = pred[i]
            #    compare_Y[test_index[i]] =  Y_scaled_test[i]
            
            #process the predicted values
            pred=pred.flatten()
            pred= pred.reshape(np.product(pred.shape),)
            pred = descale_data(matrix=pred, deviation=Y_train_deviation, ddof=ddof)
            pred = pred+Y_train_mean
            for i in range(len(test_index)):
                weird_Y[test_index[i]] = pred[i]
            pred = round_num(pred)

            for i in range(len(test_index)):
                new_Y[test_index[i]] = pred[i]

            #compute the cross-validation cumulative error (squeared error)
            err_temp = mean_squared_error(y_true=Y_test, y_pred=pred)

            #get and plot ROC curve, if want
            fpr, tpr, auc_temp = get_roc_auc(labels=Y_test, predictions=pred)        
            if i%10==0 and verbose:
                plot_roc_curve(fpr=fpr, tpr=tpr, auc=auc_temp)
                
            auc = auc + auc_temp            
            error = error + err_temp #compute square error
            train_error =train_error+train_error_temp
            train_auc = train_auc+train_auc_temp
    
    #print "Y new: ", new_Y 
    print "Y predicted scores: ", weird_Y
    print "Y predicted classes: ", new_Y
    print "Y actual labels: ", Y
    R= r2_score(Y, new_Y) 
    acc =  accuracy_score(Y, new_Y)
    F = f1_score(Y, new_Y, average='macro')
    error = float(error)/(num_repeats*num_subj) #mean error for cross-validation
    auc = auc/(num_repeats*num_folds)     #mean AUC score for cross validation
    train_error = float(train_error)/(num_repeats*num_subj)
    train_auc = float(train_auc)/(num_repeats*num_folds) 
    Q = 1 - error/(sum(Y*Y)/float(num_subj))
    if verbose:
        print "Mean cross-validation error: ", error
        print "Mean cross-validation AUC: ", auc
        print "Cross-validation Q: ", Q
    if verbose_train:
        print "Mean train cross-validation error: ", error
        print "Mean train cross-validation AUC: ", auc
        
    return error, auc, Q, R, acc, F,  train_error, train_auc


# In[ ]:


def split_between_within_subject_matrix(X, subj, unique_subj, num_unique_subj, num_subj): #SPLIT MATRIX 
    Xb = np.zeros(X.shape) #initialization
    Xw = np.zeros(X.shape)
    means = np.zeros((num_unique_subj, X.shape[1]))
    for j in range(num_unique_subj): #go through all unique subjects
        #find indexes of all entries for a certain subject (unique_subj[j])
        idx = np.where(np.array(subj)==unique_subj[j]) 
        #calculate mean for each subject unique_subj[j]      
        means[j] = np.mean(X[idx[0]], axis=0)     
    for i in range(num_subj): #go through subjects of all entries
        #find the index of the subject corresponding to subj[i] in unique_subj
        k = np.where(unique_subj==subj[i])
        #create a matrix where all entries for subject = mean (between subject variation) 
        Xb[i] = means[k[0][0]]                  
    Xw = X - Xb #get the within subjects matrix             
    return Xb, Xw


# In[ ]:


#PERMUTATION
def validate_permutation(X, Y, subj, unique_subj, num_subj, num_unique_subj, num_folds, num_repeats, scaling, 
                         num_permutations, par, verbose, ddof, method, feature_selection, num_features, 
                         step, model_params, verbose_train):
    
    err       = [] #initialization
    auc       = []
    Q         = []
    train_err = []
    train_auc = []
    
    for i in range(num_permutations):#perform permutations as many times as specified
        Y_temp = Y.copy()  #create temporary labels which then shuffle/permutate, so original is left untouched
        
        np.random.shuffle(Y_temp) #randomly shuffle the Y(labels) vector. Checks if your results will be similar
        
        #perform cross-validation for each permutation and get an array of accuracies when permutated
        err_temp, auc_temp, Q_temp, train_error_temp, train_auc_temp  = cross_validation(X=X, Y=Y_temp, subj=subj, 
                                                                            unique_subj=unique_subj,
                                                                            num_subj=num_subj, 
                                                                            num_unique_subj=num_unique_subj, 
                                                                            num_folds=num_folds, 
                                                                            num_repeats=num_repeats, 
                                                                            scaling=scaling, 
                                                                            par=par, 
                                                                            verbose=verbose,ddof=ddof,method=method, 
                                                                            feature_selection=feature_selection, 
                                                                            num_features=num_features, step=step, 
                                                                            model_params=model_params, 
                                                                            verbose_train=verbose_train)
        
        #save all the error and metrics values
        err.append(err_temp)
        auc.append(auc_temp)
        Q.append(Q_temp)
        train_err.append(train_err_temp)
        train_auc.append(train_auc_temp)
        
    if len(err) >0 and verbose :
        print "Mean permutated squared error : ", np.mean(err)
        print "Mean permutated auc : ", np.mean(auc)
        print "Mean permutated Q: ", np.mean(Q)
    elif verbose:
        print "No permutations performed"
    return err, auc, Q, train_err, train_auc


# In[ ]:


def read_data(file_name=None, mode='delta1'):
    if file_name is None: #if no input file specified, make up data
        num_subj = 30
        num_feat = 1000
        X_out = np.random.rand(num_subj, num_feat)
        for i in range(3*num_subj):
            if i%2==0:
                X[i, 0] = X[i, 0]+(np.random.rand()+0.75)*15
                X[i, 3] = X[i, 3]+(np.random.rand()+0.75)*20
                X[i, 6] = X[i, 6]+(np.random.rand()+0.75)*17
        Y =    [1 if i%2 == 0 else 0 for i in range(3*num_subj)]
        subjects = [1+i%num_subj for i in range(3*num_subj)]

    else:
        data = pd.read_excel(file_name) #read the excel file
      
        X = {}
        subjects = {}
        Y = {}
        
        #take the matabolite matrix, transpose, convert to int
        X['all'] = data.values[7:, :-9].transpose().astype(np.int64)  
        #take relevant values as labels, convert to int
        Y['all'] = (data.head().iloc[-3].values[0:148]=='risk').astype(np.int64)
        #take relevant values as subject id's, convert to int
        subjects['all'] = data.head().iloc[-2].values[0:148].astype(np.int64)
        
        unique_subj = np.unique(subjects['all']) 
        num_unique_subj = len(unique_subj)
        num_subj = len(subjects['all'])
        subj_time = {}
        mask_temp = np.array([subjects['all'][index+1]-subjects['all'][index] for index in range(num_subj-1)])
        lines = np.where(mask_temp<0)[0]
        X['delta1'] = []
        X['delta2'] = []
        X['delta3'] = []
        subjects['delta1'] = []
        subjects['delta2'] = []
        subjects['delta3'] = []
        Y['delta1'] = []
        Y['delta2'] = []
        Y['delta3'] = []
        for s in unique_subj:
            indeces_temp = np.where(subjects['all']==s)[0]
            subj_time[s]= {'t0':None, 't1':None, 't2':None}
            for el in indeces_temp:
                    if el>lines[1]:
                        subj_time[s]['t2']= el
                    elif el <= lines[0]:
                        subj_time[s]['t0']= el
                    else:
                        subj_time[s]['t1']= el
            if (subj_time[s]['t1'] is not None) and (subj_time[s]['t0'] is not None):
                X_temp = (X['all'][subj_time[s]['t1']]-X['all'][subj_time[s]['t0']]).reshape(1, X['all'].shape[1])
                X['delta1'].append(X_temp)
                subjects['delta1'].append(s)
                Y['delta1'].append(Y['all'][indeces_temp[0]])
            if (subj_time[s]['t2'] is not None) and (subj_time[s]['t1'] is not None):
                X_temp = (X['all'][subj_time[s]['t2']]-X['all'][subj_time[s]['t1']]).reshape(1, X['all'].shape[1])
                X['delta2'].append(X_temp)
                subjects['delta2'].append(s)
                Y['delta2'].append(Y['all'][indeces_temp[0]])
            if (subj_time[s]['t2'] is not None and subj_time[s]['t0'] is not None):
                X_temp = (X['all'][subj_time[s]['t2']]-X['all'][subj_time[s]['t0']]).reshape(1, X['all'].shape[1])
                X['delta3'].append(X_temp)
                subjects['delta3'].append(s)
                Y['delta3'].append(Y['all'][indeces_temp[0]])
        X['delta1'] = np.vstack( X['delta1'])
        X['delta2'] = np.vstack( X['delta2'])
        X['delta3'] = np.vstack( X['delta3'])
        
        loc = (file_name)

        # To open Workbook
        wb = xlrd.open_workbook(loc)
        sheet = wb.sheet_by_index(0)

        # For row 0 and column 0
        IDs = np.array(sheet.col_values(3)[8:])
        NetCalc = np.array(sheet.col_values(6)[8:])
        frQ = np.array(sheet.col_values(9)[8:])
        #print "The matrix we will work with is: ", X[mode]
         
    return X[mode], np.array(Y[mode]), np.array(subjects[mode]), IDs, NetCalc, frQ


# In[ ]:


#read params from command line

parser = ArgumentParser()
parser.add_argument("-c", dest="C", default=1, type=float)
parser.add_argument("-eps", dest="epsilon", default=0.1, type=float)
parser.add_argument("-ker", dest="kernel", default='rbf')
parser.add_argument("-gam", dest="gamma", default=0.0001, type=float)
parser.add_argument("-deg", dest="degree", default=3, type=int)
parser.add_argument("-fs", dest="feature_selection", default=None)
parser.add_argument("-step", dest="step", default=0.3, type=float)
parser.add_argument("-nfe", dest="num_features", default=150, type=int)
parser.add_argument("-v", dest="verbose", default=False, type=bool)
parser.add_argument("-m", dest="method", default='svm')
parser.add_argument("-vt", dest="verbose_train", default=False, type=bool)
parser.add_argument("-fn", dest="file_name", default='OGTT_INPUT.xlsx')
parser.add_argument("-mtr", dest="matrix", default='between')
parser.add_argument("-nc", dest="num_comp", default=1, type=int)
parser.add_argument("-nt", dest="num_trees", default=200, type=int)
parser.add_argument("-md", dest="max_depth", default=4, type=int)
parser.add_argument("-mf", dest="max_features", default=1, type=int)
parser.add_argument("-msl", dest="min_samples_leaf", default=1, type=int)
parser.add_argument("-nr", dest="num_repeats", default=1, type=int)
parser.add_argument("-np", dest="num_permutations", default=0, type=int)
parser.add_argument("-nf", dest="num_folds", default=20, type=int)
parser.add_argument("-sc", dest="scaling", default=True, type=bool)
parser.add_argument("-ddf", dest="ddof", default=1, type=int)
parser.add_argument("-mod", dest="mode", default='all')

args = parser.parse_args()

#run script with parameters
results = []

model_params = {}
for mod in ['rf', 'svm', 'pls', 'lda', 'nn']:
    model_params[mod] = {}
    model_params[mod]['n_trees'] = None
    model_params[mod]['max_depth'] = None
    model_params[mod]['max_features'] = None
    model_params[mod]['min_samples_leaf'] = None
    model_params[mod]['C'] = None
    model_params[mod]['epsilon'] = None
    model_params[mod]['kernel'] = None
    model_params[mod]['gamma'] = None
    model_params[mod]['degree'] = None
    model_params[mod]['n_comp'] = None

model_params['rf']['n_trees'] = args.num_trees
model_params['rf']['max_depth'] = args.max_depth 
model_params['rf']['max_features'] = args.max_features
model_params['rf']['min_samples_leaf'] = args.min_samples_leaf
model_params['svm']['C'] = args.C
model_params['svm']['epsilon'] = args.epsilon
model_params['svm']['kernel'] = args.kernel
model_params['svm']['gamma'] = args.gamma
model_params['svm']['degree'] = args.degree
model_params['pls']['n_comp'] = args.num_comp
model_params['lda']['n_comp'] = args.num_comp

betas = {'all':[], 'delta1':[], 'delta2':[], 'delta3':[], 'within':[], 'between':[], 'ID':[]}
modes = ['delta1', 'all', 'between', 'within', 'delta2', 'delta3']
#modes = ['delta1']

#num_components = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#epsilons = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
#c_values = [1, 10, 100, 1000, 10000, 100000]

for m in modes:
    if m=='all':
        args.matrix = 'all'
        args.mode = 'all'
    elif m=='between':
        args.matrix = 'between'
        args.mode = 'all'
    elif m=='within':
        args.matrix = 'within'
        args.mode = 'all'
    else:
        args.mode = m
        args.matrix = 'all'
    #print "Now the matrix is: ", args.matrix
    #print "Now the mode is: ", args.mode
    results_temp, coef,  IDs = full_script(args.num_folds,args.num_repeats,args.scaling, args.num_permutations, 
                                           args.matrix, args.file_name, args.verbose, args.ddof, args.method, 
                                           args.feature_selection, args.num_features, args.step, 
                                           model_params[args.method], args.verbose_train, args.mode)
    results_temp['mode'] = m
    print "Results: ", results_temp
    betas[m] = coef
    
    #print "???Mean of new coeff: ", np.mean(coef)
    #print "???Standard deviation of coeff: ", np.std(coef)
    #print "???Max of new coeff: ", np.amax(coef)
    #print "???Min of new coeff: ", np.amin(coef)

    #print "&&&Mean of new coeff: ", np.mean(betas[m])
    #print "&&&Standard deviation of coeff: ", np.std(betas[m])
    #print "&&&Max of new coeff: ", np.amax(betas[m])
    #print "&&&Min of new coeff: ", np.amin(betas[m])
    sys.stdout.flush()
    results.append(results_temp)

#print "OOOOOOOOOO Mean of new coeff: ", np.mean(betas.values())
#print "OOOOOOOOOO Standard deviation of coeff: ", np.std(betas.values())
#print "OOOOOOOOOO Max of new coeff: ", np.amax(betas.values())
#print "OOOOOOOOOO Min of new coeff: ", np.amin(betas.values())

betas_t = betas
betas_t['ID']= IDs

dest_file = "coef_pls_"+str(args.num_comp)+".csv"
with open(dest_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(betas_t.keys())
    writer.writerows(zip(*betas_t.values()))

res_file = "res2_"+args.method +"_"+str(args.num_features)+"_"+str(args.feature_selection)+"_"+str(args.step)+".csv"

with open(res_file, 'w') as csvfile:
    fieldnames = ['crossval_auc', 'crossval_error', 'crossval_Q', 'crossval_train_auc', 'crossval_train_err',
                  'method', 'feature_selection', 'num_features', 'step', 'n_trees', 'max_depth', 'max_features', 
                  'num_comp', 'C', 'epsilon', 'kernel', 'gamma', 'degree', 'mode']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        results_t = { your_key: r[your_key] for your_key in fieldnames}
        writer.writerow(results_t)

