import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random

from support import report, start,jsontodict, newfigure
from sklearn.cross_validation import train_test_split

# Requires directory ./report for saving images


#--------------------------
#   LOADING DATA
#--------------------------

def load_data():
    t = start("loading data")
    
    datafile = 'ItemPairs_train_merged_20000.csv'
    data = pd.read_csv(datafile)
    
    report(t, nitems=data.shape[0])
    
    return data
    
#--------------------------
# PRE PROCESSING
#--------------------------
    
def preprocess_data(data):
    t = start("preprocessing data")
    
    # Simple pre processing
    for c in data.columns:
        if data[c].dtype=='int64':
            data[c].fillna(0,inplace=True)
        elif data[c].dtype=='float64':
            data[c].fillna(0,inplace=True)
        elif data[c].dtype=='object':
            data[c].fillna("",inplace=True)
            
    
    # Remove a couple of extreme precies
    data.loc[data['price_1']>1e7,'price_1'] = 1e7
    data.loc[data['price_2']>1e7,'price_2'] = 1.1e7
    
    # Change image array to array
    data['images_array_1'] = data['images_array_1'].apply(lambda x: x.split(', '))
    data['images_array_2'] = data['images_array_2'].apply(lambda x: x.split(', '))
    data['images_len1'] = data['images_array_1'].apply(lambda x: len(x))
    data['images_len2'] = data['images_array_2'].apply(lambda x: len(x))
    
    # Title bag of words
    data['title_bag_len_1'] = data['title_1'].apply(lambda x: len(x.split(' ')))
    data['title_bag_len_2'] = data['title_2'].apply(lambda x: len(x.split(' '))) 
    
    # Description bag of words
    data['description_bag_len_1'] = data['description_1'].apply(lambda x: len(x.split(' ')))
    data['description_bag_len_2'] = data['description_2'].apply(lambda x: len(x.split(' '))) 
    
    # Convert JSON attr to dictionaries
    data['attrsJSON_1'] = data['attrsJSON_1'].apply(lambda x: jsontodict(x))
    data['attrsJSON_2'] = data['attrsJSON_2'].apply(lambda x: jsontodict(x))
    data['attrs_len1'] = data['attrsJSON_1'].apply(lambda x: len(x))
    data['attrs_len2'] = data['attrsJSON_2'].apply(lambda x: len(x))
    
    report(t, nitems=data.shape[0])
    
    return data
    
#--------------------------
# FEATURE ENGINEERING
#--------------------------
    
def calculate_features(data):
    t = start("calculating features")
    # Store old column in order to only pass on calculated new feature columns    
    old_cols = data.columns    
    if 'isDuplicate' in data.columns:
        old_cols = data.columns.drop(['isDuplicate'])
        
    # Title
    data['title_equal'] = np.equal(data['title_1'], data['title_2'])
    data['title_len_max'] = np.maximum(data['title_1'].str.len(), data['title_2'].str.len())
    data['title_bag_len_max'] = np.maximum(data['title_bag_len_1'], data['title_bag_len_2'])
    
    # Category
    data['categoryID'] = data['categoryID_1']
    
    # Description
    data['description_equal'] = np.equal(data['description_1'],data['description_2'])
    data['description_len_max'] = np.maximum(data['description_bag_len_1'], data['description_bag_len_2'])
    data['description_len_diff'] = np.abs(np.subtract(data['description_1'].str.len(),data['description_2'].str.len()))
    data['description_bag_len_diff'] = np.abs(np.subtract(data['title_bag_len_1'],data['description_bag_len_2']))
    
    # LocationID & Coordinates
    data['dist'] = np.abs( data['lat_1'] - data['lat_2']) + np.abs( data['lon_1'] - data['lon_2'])
    data['metroID_equal'] = np.equal(data['metroID_1'],data['metroID_2'])
    data['locationID_equal'] = np.equal(data['locationID_1'],data['locationID_2'])
    
    # images_array
    data['images_equal'] = np.equal(data['images_array_1'],data['images_array_2'])
    data['images_len_max'] = np.maximum(data['images_len1'], data['images_len2'])
    data['images_len_diff'] = np.subtract(data['images_len1'], data['images_len2'])
    
    # JSON attributes
    data['attrs_equal'] = np.equal(data['attrsJSON_1'],data['attrsJSON_2'])
    data['attrs_len_equal'] = np.equal(data['attrs_len1'],data['attrs_len2'])
    data['attrs_len_max'] = np.maximum(data['attrs_len1'],data['attrs_len2'])
    
    # Features based on price properties
    data['price_equal'] = np.equal(data['price_1'],data['price_2'])
    data['price_max'] = np.maximum(data['price_1'],data['price_2'])
    
    X = data.drop(old_cols, axis=1)
    
    if 'isDuplicate' in X:
        y = data.isDuplicate
        X = X.drop('isDuplicate', axis=1)
    else:
        # If calculating features of test data set
        y = None
        
    report(t, nitems=data.shape[0])
        
    return (X,y)
#--------------------------
# SUPPORT FUNCTIONS
#--------------------------

from sklearn import tree, metrics, grid_search, cross_validation
def get_decisiontree_classifier(X_train, y_train, params=None):
    param_grid = {"criterion": ["gini", "entropy"],
                  "max_depth": [2,3,4,5,6,7,8]}
    
    if params is None:
    
        tr = tree.DecisionTreeClassifier()
        
        t = start("training decision tree")
        cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10,test_size=0.3, random_state=123)
        clf = grid_search.GridSearchCV(tr, param_grid, cv=cv, n_jobs=4, scoring='roc_auc')
        clf = clf.fit(X_train,y_train)
        report(t, nitems=10*len(param_grid))
        
        print("Best score:{} with scorer {}".format(clf.best_score_, clf.scorer_))
        print "With parameters:"
    
        best_parameters = clf.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print '\t%s: %r' % (param_name, best_parameters[param_name]) 
    else:
        clf = tree.DecisionTreeClassifier(**params)
        clf = clf.fit(X_train,y_train)
        
    return clf
    
from sklearn.neighbors import KNeighborsClassifier
def get_knn_classifier(X_train, y_train, params=None):
    param_grid = {'weights': ['uniform', 'distance'],
                  'n_neighbors': [5, 10, 20, 50, 100, 200, 500]}
                  
    if params is None:
                  
        knn = KNeighborsClassifier()
        t = start("training knn")
        cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10,test_size=0.2, random_state=123)
        clf = grid_search.GridSearchCV(knn, param_grid, cv=cv, n_jobs=4, scoring='roc_auc')
        clf = clf.fit(X_train,y_train)
        report(t, nitems=10*len(param_grid))
        
        print("Best score:{} with scorer {}".format(clf.best_score_, clf.scorer_))
        print "With parameters:"
    
        best_parameters = clf.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print '\t%s: %r' % (param_name, best_parameters[param_name]) 
    else:
        clf = KNeighborsClassifier(**params)
        clf = clf.fit(X_train,y_train)
        
    return clf

from sklearn.linear_model import LogisticRegression
def get_logistic_classifier(X_train, y_train, params=None):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1','l2']}
                  
    if params is None:
                  
        log = LogisticRegression()
        t = start("training logistic ")
        cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10,test_size=0.2, random_state=123)
        clf = grid_search.GridSearchCV(log, param_grid, cv=cv, n_jobs=4, scoring='roc_auc')
        clf = clf.fit(X_train,y_train)
        report(t, nitems=10*len(param_grid))
        
        print("Best score:{} with scorer {}".format(clf.best_score_, clf.scorer_))
        print "With parameters:"
    
        best_parameters = clf.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print '\t%s: %r' % (param_name, best_parameters[param_name]) 
    else:
        clf = LogisticRegression(**params)
        clf = clf.fit(X_train,y_train)
        
    return clf


from sklearn.ensemble import RandomForestClassifier
def get_randomforest_classifier(X_train, y_train, params=None):
    param_grid = {"max_depth": [4, 5, 6, 7],
                  "max_features": [3, 5],
                  "criterion": ["gini", "entropy"]}
                  
    if params is None:
                  
        log = RandomForestClassifier()
        t = start("training random forest ")
        cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10,test_size=0.2, random_state=123)
        clf = grid_search.GridSearchCV(log, param_grid, cv=cv, n_jobs=4, scoring='roc_auc')
        clf = clf.fit(X_train,y_train)
        report(t, nitems=10*len(param_grid))
        
        print("Best score:{} with scorer {}".format(clf.best_score_, clf.scorer_))
        print "With parameters:"
    
        best_parameters = clf.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print '\t%s: %r' % (param_name, best_parameters[param_name]) 
    else:
        clf = RandomForestClassifier(**params)
        clf = clf.fit(X_train,y_train)
        
    return clf
    
from xgboost.sklearn import XGBClassifier 
#import xgboost as xgb 
def get_xgboost_classifier(X_train, y_train, X_val, y_val,params=None):
    
    param_grid = {'max_depth':[3,5,7], 'min_child_weight': [1,3,5], 'n_estimators': 50}
    
    if params is None:
        xgb = XGBClassifier(
                 learning_rate =0.2,
                 objective= 'binary:logistic',
                 seed=27)
                 
        t = start("training xgboost ")
        cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10,test_size=0.2, random_state=123)
        clf = grid_search.GridSearchCV(xgb, param_grid, cv=cv, n_jobs=1, scoring='roc_auc')
        clf = clf.fit(X_train,y_train)
        report(t, nitems=10*len(param_grid))
        
        print("Best score:{} with scorer {}".format(clf.best_score_, clf.scorer_))
        print "With parameters:"
    
        best_parameters = clf.best_estimator_.get_params()
        for param_name in sorted(param_grid.keys()):
            print '\t%s: %r' % (param_name, best_parameters[param_name]) 
    else:
        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train, eval_set =  [(X_train,y_train),(X_val,y_val)], eval_metric='auc', verbose=False)
        
    if plot_cv_curves:
        train = clf.evals_result()['validation_0']['auc']
        val = clf.evals_result()['validation_1']['auc']
        
        plot_cv_curve(train, val, params['learning_rate'])
        
    if plot_feature_importance:
        plot_feature_importance(clf,params['learning_rate'])
            
    return clf
        

        

def report_result(clf, X_test, y_test):
    y_pred = clf.predict_proba(X_test)[:,1]
    print ""
    print "{} AUC Score: {:.4f}".format(type(clf),metrics.roc_auc_score(y_test,y_pred))
   

    #print "Classification report:"    
    #print metrics.classification_report(y_test,y_pred,target_names=['Original','Duplicate'])
    
    #print "Confusion matrix:"    
    #print pd.DataFrame(metrics.confusion_matrix(y_test,y_pred,labels=[0,1]),columns=['Original','Duplicate'],index=['Original','Duplicate'])

    return metrics.roc_auc_score(y_test,y_pred)
    
def plot_cv_curve(test_scores, train_scores, identifier):
    newfigure("Progress AUC score during boosting " + str(identifier))
    plt.plot(test_scores,'g',label='Validation set')
    plt.plot(train_scores,'r',label='Train set')
    plt.grid()
    plt.xlabel('Boosting round')
    plt.ylabel('AUC Score')
    plt.legend(loc=4)  
    plt.savefig('report/xgb_eval_curves_' + str(identifier) + '.png')
    
def plot_feature_importance(clf, identifier):
    
    feat_imp = pd.Series(clf.booster().get_fscore()).sort_values(ascending=False)

    newfigure('Feature Importances'+str(identifier))
    feat_imp.plot(kind='bar')
    plt.ylabel('Feature Importance Score')
    plt.grid()
    plt.tight_layout()
    plt.savefig('report/xgb_feature_importance_' + str(identifier) + '.png')
    
#--------------------------
# SUPPORT FUNCTIONS
#--------------------------
def run_case(case):
    name = case[0]
    X_train = case[1]
    y_train = case[2]
    X_val = case[3]
    y_val = case[4]
    clfparams = case[5]
    classifier = case[6]
    
    print("Running case: %s" % name)
    
    if classifier=="xgb":
        return get_xgboost_classifier(X_train, y_train, X_val, y_val, clfparams)
    elif classifier=="knn":
        return get_knn_classifier(X_train, y_train, clfparams)
    elif classifier=="log":
        return get_logistic_classifier(X_train, y_train, clfparams)
    elif classifier=="rdf":
        return get_randomforest_classifier(X_train, y_train, clfparams)
    elif classifier=="tree":
        return get_decisiontree_classifier(X_train, y_train, clfparams)
        
def run_cases(cases, groupName):
    
    print("Running %s" % groupName)    
    print("")
    
    names = [x[0] for x in cases]
    
    scores = []
    clfs = []
    times_t = []
    times_p = []
    
    for c in cases:
        
        X_val = c[3]
        y_val = c[4]
        
        s = time.time()      
        
        # Train model
        clf = run_case(c)
        clfs.append( clf )
        times_t.append( time.time()-s )
        
        # Predict
        s = time.time()  
        scores.append( report_result(clf, X_val, y_val) )
        times_p.append( time.time()-s )
        
    width = 0.5    
    
    newfigure(groupName)
    plt.bar(np.arange(len(cases)),scores, width)
    rotation='vertical' if len(names[0])>5 else 'horizontal'
    plt.xticks(np.arange(len(cases))+width/2, names, rotation=rotation)
    plt.ylabel("AUC Score")
    plt.ylim([0.5, 1])
    plt.xlim([-width,len(cases)])
    plt.tight_layout()
    plt.grid()
    
    ax = plt.gca()
    rects = ax.patches
    for rect, label in zip(rects, scores):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, "{:.4f}".format(label), ha='center', va='bottom')
        
    plt.savefig('report/%s_auc.png' % groupName)
    
    if plot_memory_usage:
        # Approximate memory usage by pickle dump
        import pickle
        memsizes = [float(len(pickle.dumps((x)))) for x in clfs]
        
        newfigure(groupName)
        plt.bar(np.arange(len(cases)),memsizes, width)
        plt.xticks(np.arange(len(cases))-1+width/2, names, rotation='vertical')
        plt.ylabel("Memory size (kB)")
        plt.tight_layout()
        plt.xlim([-width,len(cases)])
        plt.grid()
        plt.title(groupName)
        plt.savefig('report/%s_mem.png' % groupName)
    
    if plot_training_time:
        newfigure(groupName)
        plt.bar(np.arange(len(cases)),np.multiply(times_t,1000), width)
        plt.xticks(np.arange(len(cases))+width/2, names, rotation='vertical')
        plt.ylabel("Training time (ms)")
        plt.xlim([-width,len(cases)])
        plt.tight_layout()
        plt.grid()
        plt.savefig('report/%s_training_time.png' % groupName)
    
    if plot_prediction_time:
        newfigure(groupName)
        plt.bar(np.arange(len(cases)),np.multiply(times_p,1000), width)
        plt.xticks(np.arange(len(cases))+width/2, names, rotation='vertical')
        plt.ylabel("Prediction time (ms)")
        plt.xlim([-width,len(cases)])
        plt.tight_layout()
        plt.grid() 
        plt.savefig('report/%s_pred_time.png' % groupName)
    
    return (scores, clfs, times_t, times_p)
#--------------------------
# CASES FUNCTIONS
#--------------------------

def compare_basic_classifiers():
    cases = []
    
    optimized_tree_params = {"criterion": "gini", "max_depth": 7}
    cases.append( ("Tree" , X_train, y_train, X_val, y_val, optimized_tree_params, "tree") )
    optimized_knn_params = {'weights': 'distance','n_neighbors': 100}
    cases.append( ("KNN" , X_train, y_train, X_val, y_val, optimized_knn_params, "knn") )
    optimized_log_params = {'C': 0.1,'penalty': 'l1'}
    cases.append( ("LOG" , X_train, y_train, X_val, y_val, optimized_log_params, "log") )
    
    return run_cases(cases,"Comparison basic classifiers")

def compare_tree_ensembles():
    cases = []
    
    optimized_tree_params = {"criterion": "entropy", "max_depth": 7}
    cases.append( ("Tree" , X_train, y_train, X_val, y_val, optimized_tree_params, "tree") )
    optimized_randomforest_params = {"criterion": "entropy", "max_depth": 7,'max_features': 5}
    cases.append( ("Rdf" , X_train, y_train, X_val, y_val, optimized_randomforest_params, "rdf") )
    optimized_xgboost_params = {'max_depth': 7,'min_child_weight': 3}
    cases.append( ("XGB" , X_train, y_train, X_val, y_val, optimized_xgboost_params, "xgb") )
    
    return run_cases(cases,"Comparison tree ensembles")

def compare_tree_ensembles_gridsearch():
    cases = []
    
    cases.append( ("Tree" , X_train, y_train, X_val, y_val ,None, "tree") )
    cases.append( ("Rdf" , X_train, y_train, X_val, y_val , None, "rdf") )
    cases.append( ("XGB" , X_train, y_train, X_val, y_val, None, "xgb") )
    
    return run_cases(cases,"Comparison tree ensembles (optimized)")

def compare_xgboost_hyperparams_maxdepth():
    cases = []
    
    
    pars = {'n_estimators': 100,'max_depth': 4,'min_child_weight': 5}
    cases.append( ("4" , X_train, y_train, X_val, y_val ,pars, "xgb") )
    pars = {'n_estimators': 100,'max_depth': 6,'min_child_weight': 5}
    cases.append( ("6" , X_train, y_train, X_val, y_val , pars, "xgb") )
    pars = {'n_estimators': 100,'max_depth': 8,'min_child_weight': 5}
    cases.append( ("8" , X_train, y_train, X_val, y_val, pars, "xgb") )
    
    return run_cases(cases,"Comparison XGBoost max-depth")
    
def compare_xgboost_hyperparams_minchildweight():
    cases = []
    
    pars = {'n_estimators': 100,'max_depth': 6,'min_child_weight': 5}
    cases.append( ("5" , X_train, y_train, X_val, y_val ,pars, "xgb") )
    pars = {'n_estimators': 100,'max_depth': 6,'min_child_weight': 15}
    cases.append( ("15" , X_train, y_train, X_val, y_val , pars, "xgb") )
    pars = {'n_estimators': 100,'max_depth': 6,'min_child_weight': 25}
    cases.append( ("25" , X_train, y_train, X_val, y_val, pars, "xgb") )
    
    return run_cases(cases,"Comparison XGBoost min child weigth")
    
def compare_xgboost_hyperparams_etanestimator():
    cases = []
    
    pars = {'n_estimators': 50,'learning_rate':0.3,'max_depth': 6,'min_child_weight': 3}
    cases.append( ("0.3" , X_train, y_train, X_val, y_val ,pars, "xgb") )
    pars = {'n_estimators': 100,'learning_rate':0.2,'max_depth': 6,'min_child_weight': 5}
    cases.append( ("0.2" , X_train, y_train, X_val, y_val , pars, "xgb") )
    pars = {'n_estimators': 150,'learning_rate':0.1,'max_depth': 6,'min_child_weight': 7}
    cases.append( ("0.1" , X_train, y_train, X_val, y_val, pars, "xgb") )
    
    return run_cases(cases,"Comparison XGBoost max-depth")

#--------------------------
# MAIN FUNCTION
#--------------------------       
if __name__ == "__main__":
    
    
    
    rs = sum([ord(x) for x in "hello reviewer"])
    show_feature_importance = True
    plot_memory_usage = False
    plot_prediction_time = True
    plot_training_time = True
    plot_cv_curves = True
    
    sample_data = load_data()
    
    preproc_data = preprocess_data(sample_data)
    
    (X,y) = calculate_features(preproc_data)
    
    X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.2, random_state=rs, stratify=y)
    
    plt.close('all')
    
    
    compare_basic_classifiers()
    # compare_tree_ensembles
    # compare_tree_ensembles_gridsearch
    # compare_xgboost_hyperparams_maxdepth
    # compare_xgboost_hyperparams_minchildweight
    # compare_xgboost_hyperparams_etanestimator()

    
