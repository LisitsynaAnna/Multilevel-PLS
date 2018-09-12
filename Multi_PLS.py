
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
    print "Y_temp: ", Y_temp

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
        print "pred_temp before descaling and decentering: ", pred_temp 
        if feature_selection != 'lda' and scaling:
            pred_temp = descale_data(matrix=pred_temp, deviation=deviation, ddof=ddof ) #descale
            pred_temp = pred_temp+mean #add mean
        elif feature_selection != 'lda':
            pred_temp = pred_temp+mean
        print "pred_temp after descaling and decentering: ", pred_temp 
        pred_temp = round_num(pred_temp)

        train_error = mean_squared_error(y_true=Y_temp, y_pred=pred_temp)
        fpr, tpr, auc = get_roc_auc(labels=Y_temp, predictions=pred_temp)#AUC if needed

        if verbose_train:
            print "Training error: ", train_error/num_samples
            print "Training error computed in library: ", train_error_temp
            print "Training auc: ", auc
            plot_roc_curve(fpr, tpr, auc)
        print "coefficients(betas?): ", coeff[:10]
    
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
    return features, model, train_error, auc, model.coef_

def steps_simple_classifier(X, Y, subj, ids, num_samples, mean,deviation, ddof, method, feature_selection, model_params,
               num_features, step, verbose_train, scaling, num_iter):
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
    coeff = model_temp.coef_
    print "coeffs: ", model_temp.coef_
    print "coeffs: ", np.shape(model_temp.coef_)
    print "new number: ", np.rint(X.shape[1]/2)
    print "type of new number: ", type(np.rint(X.shape[1]/2))
    #features = np.argpartition(model_temp.coef_.reshape(np.shape(model_temp.coef_)[0],), int(np.rint(X.shape[1]/2)))[-1*int(np.rint(X.shape[1]/2)):]

    #rfe = RFE(estimator=model_temp, n_features_to_select=num_features, step=step)
    #fit = rfe.fit(X, Y)
    #features = fit.support_
    X_new =X
    new_ids = ids
    new_features = features
    new_coeff = coeff
    print "new_coeff starting", new_coeff[:10]
    print "length new_coeff ", len(new_coeff)
    for i in range(num_iter):
    	new_model_temp = copy.copy(model)
    	new_features = np.argpartition(new_coeff.reshape(np.shape(new_coeff)[0],), int(np.rint(X_new.shape[1]/2)))[-1*int(np.rint(X_new.shape[1]/2)):]
    	new_ids = new_ids[new_features]
    	X_new = X_new[:, new_features]
    	new_model_temp.fit(X, Y)
    	new_coeff = new_model_temp.coef_
    	print "new_coeff ", new_coeff[:10]
    	print "length new_coeff ", len(new_coeff)
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
    print "new_coeff last", new_coeff[:10]
    print "length new_coeff ", len(new_coeff)
    return new_features, model, train_error, auc, new_coeff

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
def perform_multilevel_pls(X, Y, subj, ids,  unique_subj, num_unique_subj, num_subj, scaling, par, ddof, method, 
                           feature_selection, num_features, step, model_params, verbose_train, num_iter):
    
    Xb= np.zeros(X.shape) 
    Xw= np.zeros(X.shape)
    #mean-centering
    X_centered = center_data(matrix=X, mean=None) #mean centering of data
    Y_centered = center_data(matrix=Y, mean=None) 
    print "X_centered: ", X_centered
    print "Y_centered: ", Y_centered

    #split matrix into between and within subject variations
    if par!='all':
        Xb, Xw = split_between_within_subject_matrix(X=X_centered, subj=subj, unique_subj=unique_subj, 
                                                 num_unique_subj=num_unique_subj, num_subj=num_subj)
    	print "Xb: ", Xb
    	print "Xw: ", Xw

    #scaling (if necessary and specified)
    if scaling: #scale the data, if the flag is true
        X_scaled = scale_data(matrix=X_centered, deviation=None, ddof=ddof)
        Y_scaled = scale_data(matrix=Y_centered, deviation=None, ddof=ddof)                      
        if par!='all':
            Xb_scaled = scale_data(matrix=Xb, deviation=None, ddof=ddof)
            Xw_scaled = scale_data(matrix=Xw, deviation=None, ddof=ddof)
            print "Xb_scaled: ", Xb_scaled
            print "Xw_scaled: ", Xw_scaled
    else:
        X_scaled = X_centered
        Y_scaled = Y_centered
        if par!='all':
            Xb_scaled = Xb
            Xw_scaled = Xw
    	
    print "Y_scaled: ", Y_scaled

    #which matrix are we interested in
    if par=='all':
        X_target = X_scaled
    elif par=='between':
        X_target = Xb_scaled
    elif par=='within':
        X_target = Xw_scaled

    print "X_target: ", X_target 
    
    #perform classifier (PLS or other) on target data
    #features, model, train_error, train_auc, coeff= classifier(X=X_target, Y=Y_scaled, subj=subj, num_samples=num_subj,
    #                                                       mean=np.mean(Y), deviation= np.std(Y, axis=0, ddof=ddof), 
    #                                                       ddof=ddof, method=method, 
    #                                                       feature_selection=feature_selection, 
    #                                                       num_features=num_features, step=step, 
    #                                                       model_params=model_params, verbose_train=verbose_train, 
    #                                                       scaling=scaling)
#
   # 
    features, model, train_error, train_auc, coeff= steps_simple_classifier(X=X_target, Y=Y_scaled, subj=subj, ids= ids, num_samples=num_subj,
                                                           mean=np.mean(Y), deviation= np.std(Y, axis=0, ddof=ddof), 
                                                           ddof=ddof, method=method, 
                                                           feature_selection=feature_selection, 
                                                           num_features=num_features, step=step, 
                                                           model_params=model_params, verbose_train=verbose_train, 
                                                           scaling=scaling, num_iter=num_iter)
    print "coeffs in perform_multilevel_pls: ", coeff[:10]
    #features, model, train_error, train_auc, coeff= simple_classifier(X=X_target, Y=Y_scaled, subj=subj, num_samples=num_subj,
    #                                                       mean=np.mean(Y), deviation= np.std(Y, axis=0, ddof=ddof), 
    #                                                       ddof=ddof, method=method, 
    #                                                       feature_selection=feature_selection, 
    #                                                       num_features=num_features, step=step, 
    #                                                       model_params=model_params, verbose_train=verbose_train, 
    #                                                       scaling=scaling)

    
    
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
                feature_selection,num_features,step, model_params,verbose_train, mode, num_iter):
    
    #READ OR MAKE UP DATA
    X, Y, subj, IDs, NetCalc, frQ = read_data(filename, mode)
    print "X: ", X

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
    crossval_R=0
    crossval_acc=0
    crossval_F=0
    

    #PERFORMING MULTILEVEL PLS ON WHOLE DATASET

    features, full_model, full_train_error, full_train_auc, Xb, Xw, Y_scaled, X_scaled, coeff= perform_multilevel_pls(X=X, Y=Y, subj=subj, ids=IDs,
                           unique_subj=unique_subj, num_subj=num_subj, 
                           num_unique_subj=num_unique_subj, scaling=scaling, par=par, 
                           ddof=ddof, method=method, feature_selection=feature_selection, 
                           num_features=num_features, step=step, model_params=model_params, verbose_train=verbose_train, num_iter=num_iter)

   
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
    #print "coeff in full_script: ", coeff[:10]
    new_coefficients = np.zeros(len(features))
    count=0
    for i in range(len(features)):
    	if features[i]:
    		new_coefficients[i] = coeff[count]
    		count = count+1
    print "new coefficients: ", new_coefficients[:10]
    #IDs = np.array(IDs)
    #coefficients = {}
    #for id in IDs:
    #    coefficients[id]=0
    #for i in range(len(IDs[features])):
    #    coefficients[IDs[i]]=coeff[i]
    ##print "coefficients in full_script: ", coefficients
    #print "coefficients.values() in full_script: ", coefficients.values()[:10]
    #print "###!Mean of new coeff: ", np.mean(coefficients.values())
    #print "###!Standard deviation of coeff: ", np.std(coefficients.values())
    #print "###!Max of new coeff: ", np.amax(coefficients.values())
    #print "###!Min of new coeff: ", np.amin(coefficients.values())
    #if verbose: 
    #    print "CROSS_VALIDATION ON ACTUAL DATA: "
        
    #CROSS-VALIDATION
    #crossval_error, crossval_auc, crossval_Q, crossval_R, crossval_acc, crossval_F, crossval_train_err, crossval_train_auc =    cross_validation(X=X, Y=Y, subj=subj, unique_subj=unique_subj, num_subj=num_subj, num_unique_subj=num_unique_subj, 
    #                 num_folds=num_folds, num_repeats=num_repeats, scaling=scaling, par=par, 
    #                 verbose=verbose, ddof=ddof, method=method, feature_selection=feature_selection, 
    #                 num_features = num_features, step=step, model_params=model_params, verbose_train=verbose_train)
 # 
    #if verbose: 
    #    print "PERMUTATED DATA CROSS_VALIDATION: "
        
    #PERMUTATE
    #permutation_error, permutation_auc, permutation_Q, permutation_train_error, permutation_train_auc =                                                        validate_permutation(X=X, Y=Y, subj=subj,
    #                                                                         unique_subj=unique_subj, 
    #                                                                         num_subj=num_subj,
    #                                                                         num_unique_subj=num_unique_subj,
    #                                                                         num_folds=num_folds,
    #                                                                         num_repeats=num_repeats, 
    #                                                                         scaling=scaling,
    #                                                                         num_permutations=num_permutations, 
    #                                                                         par=par, verbose=verbose, 
    #                                                                         ddof=ddof,method=method,
    #                                                                         feature_selection=feature_selection, 
    #                                                                         num_features=num_features, step=step,
    #                                                                         model_params=model_params, 
     #                                                                        verbose_train=verbose_train)
   
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
    
    return results, new_coefficients, IDs


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


def read_data(file_name=None, mode='delta1', use_kf=False):
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
        best_IDs = [123046, 117583, 86756, 133438, 83147, 126667,172100,142397,120639,120289,156326,122478,149189,96777,
27106,
111945,
143274,
165258,
131037,
157557,
116613,
124210,
141199,
116193,
104914,
157654,
138437,
136815,
145389,
126381,
134480,
116092,
115231,
111308,
126611,
170398,
122508,
154395,
70020,
114368,
111846,
115198,
118088,
111233,
112431,
129222,
123808,
88233,
140056,
98225,
129979,
105895,
93554,
168699,
46429,
118641,
63581,
160205,
138660,
143546,
106130,
160226,
120674,
117133,
136197,
150296,
150151,
172174,
91957,
49544,
46949,
66586,
131636,
118862,
151173,
112437,
165739,
159266,
141582,
120258,
121102,
141866,
122867,
120459,
108997,
158654,
116452,
138190,
87247,
123912,
111652,
57571,
122104,
134658,
99130,
75234,
118373,
143395,
178875,
116969,
114194,
143665,
135723,
123011,
143764,
139889,
99724,
123297,
173147,
99833,
45822,
127758,
105372,
129522,
84139,
99759,
102071,
150305,
123324,
152526,
141850,
160588,
157541,
112401,
109664,
75714,
77782,
61572,
158015,
108904,
158755,
150408,
99860,
136238,
116874,
159126,
105832,
152891,
150153,
136635,
108399,
96311,
148861,
41012,
135837,
166516,
161209,
92918,
111856,
117044,
126660,
81993,
118150,
102982,
147427,
60576,
112635,
90322,
65937,
123300,
86897,
140036,
118805,
24102,
81741,
120293,
152433,
114713,
130687,
116150,
15941,
41791,
156294,
149273,
166110,
84512,
117161,
152638,
158804,
109189,
132854,
160162,
160580,
105321,
115801,
86401,
94999,
143598,
115477,
114526,
23204,
94848,
89149,
117109,
126393,
105371,
76912,
151009,
101847,
93456,
136303,
140721,
102537,
174794,
151593,
100728,
101025,
120670,
114213,
117664,
83231,
158312,
147886,
106519,
116765,
126557,
62597,
116684,
149030,
89192,
118560,
122442,
140495,
138780,
113328,
174883,
153224,
147760,
144395,
129606,
140235,
138644,
94997,
98351,
95739,
142471,
106468,
125429,
165086,
98783,
158277,
118568,
38745,
135112,
46468,
160346,
111272,
148683,
97013,
72241,
96358,
129183,
106671,
124758,
93477,
160577,
76415,
171466,
144035,
142284,
137802,
150517,
174979,
113884,
133553,
134644,
104612,
97056,
184329,
144820,
145249,
157989,
109380,
125320,
161386,
134281,
108827,
86504,
134358,
134992,
106307,
112171,
98538,
143145,
138484,
152261,
163107,
103372,
124782,
156301,
93876,
101171,
161317,
147111,
175066,
167862,
93015,
98806,
129885,
132331,
91894,
156017,
97581,
123087,
138061,
121029,
131804,
174279,
25378,
132130,
170927,
120761,
33607,
136140,
159222,
110192,
101234,
29130,
119578,
185709,
88722,
118349,
139402,
144127,
144007,
100418,
141075,
117573,
94215,
108871,
151299,
134953,
94873,
133646,
124628,
169783,
133810,
106375,
147917,
126225,
120735,
107153,
175673,
164989,
161732,
112685,
151337,
53126,
108345,
117812,
72767,
180769,
137656,
88089,
134960,
173188,
148226,
53154,
81194,
114268,
118013,
153926,
70486,
89932,
143869,
85974,
142786,
133451,
156972,
122803,
26440,
101797,
134826,
54469,
109826,
152181,
91913,
98206,
103657,
95618,
157153,
93108,
145919,
64418,
120961,
74704,
83051,
108622,
92065,
143847,
145817,
133631,
156253,
75611,
128156,
133623,
99988,
114201,
103674,
96732,
133997,
146842,
111268,
114803,
156926,
156122,
112335,
147223,
116388,
180201,
143287,
136217,
100764,
83709,
110274,
114684,
148039,
57622,
121381,
115143,
132690,
144322,
125941,
61031,
192530,
144329,
56977,
153432,
151744,
149525,
149649,
133929,
111905,
156284,
97619,
145408,
120259,
145623,
176474,
145664,
114451,
84867,
117846,
78910,
173199,
114111,
136091,
132134,
102054,
115738,
125532,
148528,
146956,
133430,
153530,
27836,
78839,
133993,
144012,
179103,
55674,
79004,
56201,
110757,
124392,
109893,
120669,
141743,
140464,
133371,
156409,
146587,
34067,
144914,
137624,
133737,
121614,
91437,
168924,
94895,
105099,
157135,
83971,
123016,
125737,
134154,
128750,
153667,
111866,
103707,
144951,
143689,
55564,
133992,
133826,
55040,
38860,
104326,
113384,
110631,
151351,
156797,
169965,
41131,
149503,
116141,
122054,
172297,
118014,
103675,
172528,
69565,
140558,
123793,
135131,
147353,
73158,
114199,
131956,
70656,
176937,
102078,
133428]
        loc = (file_name)

        # To open Workbook
        wb = xlrd.open_workbook(loc)
        sheet = wb.sheet_by_index(0)

        # For row 0 and column 0
        IDs = np.array(sheet.col_values(3)[8:])
        NetCalc = np.array(sheet.col_values(6)[8:])
        frQ = np.array(sheet.col_values(9)[8:])
        #print "The matrix we will work with is: ", X[mode]
        new_features = [IDs[ix] in best_IDs for ix in range(len(IDs))]



        #take the matabolite matrix, transpose, convert to int
        X['all'] = data.values[7:, :-9].transpose().astype(np.int64)  
        #take relevant values as labels, convert to int
        Y['all'] = (data.head().iloc[-3].values[0:148]=='risk').astype(np.int64)
        #take relevant values as subject id's, convert to int
        subjects['all'] = data.head().iloc[-2].values[0:148].astype(np.int64)

        if use_kf:
        	X['all'] = X['all'][:, new_features]
        	IDs = best_IDs

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
parser.add_argument("-nfe", dest="num_features", default=8382, type=int)
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
#modes = ['all']
#betas = {'all':[]}

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
                                           model_params[args.method], args.verbose_train, args.mode, 4)
    results_temp['mode'] = m
    print "Coef at the end: ", coef[:10]
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

print "Super last: ", betas_t['between'][:10]
print "Super last: ", zip(*betas_t.values()).sort(key = lambda t: t[-1])[:10]

dest_file = "coef_pls_"+str(args.num_comp)+".csv"
with open(dest_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(betas_t.keys())
    writer.writerows(zip(*betas_t.values()).sort(key = lambda t: t[-1]))

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

