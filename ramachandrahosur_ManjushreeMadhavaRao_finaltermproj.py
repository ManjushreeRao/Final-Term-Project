#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Installing the required Libraries
get_ipython().system('pip install seaborn')
get_ipython().system('pip install sklearn')


# In[ ]:


import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_validate, cross_val_predict, validation_curve
from sklearn.metrics import confusion_matrix, make_scorer


# In[259]:


#Load and print data
data = pd.read_csv('./data/diabetes.csv')
data.head()


# In[260]:


data.describe()


# In[261]:


for column in list(data):
    print(f'Column name:  {column} , no of null : {data[column].size - data[column].count()}')


# In[262]:


# Diabetics.csv has 2 times non diabetic to 1 time diabetic data
data.hist(column='Outcome'))


# In[263]:


sns.heatmap(data.corr(),annot=True)
# Corelation b/w fields


# In[264]:


# features on the 'x' axis
X = data.drop("Outcome",axis = 1)

# label in 'y' axis
y = data.Outcome


# In[265]:


X.head()


# In[266]:


y.head()


# In[267]:


cv = KFold(n_splits=10, random_state=10, shuffle=True)


# In[268]:


def plot_results(train_score, test_score, title, xlabel):
    #standard deviation and mean calculated for testing and trainig scores 
    mean_train_score = np.mean(train_score, axis = 1)
    std_train_score = np.std(train_score, axis = 1)

   
    mean_test_score = np.mean(test_score, axis = 1)
    std_test_score = np.std(test_score, axis = 1)

    # Creating the Plot for above
    plt.plot(parameter_range, mean_train_score, label = "Training Score", color = 'b')
    plt.plot(parameter_range, mean_test_score, label = "Cross Validation Score", color = 'g')

    # values considered for algo comparison
    best_neighbor = parameter_range[np.argmax(mean_test_score)]

    # Creating the plot
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.show()


# In[269]:


# best n_neighbors for knn algo
parameter_range = np.arange(1, 100, 5)
 
# 10-fold cross validation
train_score, test_score = validation_curve(KNeighborsClassifier(), X, y,
                                       param_name = "n_neighbors",
                                       param_range = parameter_range,
                                        cv = cv, scoring = "accuracy")

plot_results(train_score, test_score, "Validation Curve with KNN Classifier", "Number of Neighbours")


# In[270]:


# best n_estimators in random forest 
parameter_range = np.arange(10, 1000, 100)

# 10-fold cross validation
train_score, test_score = validation_curve(RandomForestClassifier(), X, y,
                                       param_name = "n_estimators",
                                       param_range = parameter_range,
                                        cv = cv, scoring = "accuracy")
 
plot_results(train_score, test_score, "Validation Curve with RandomForest Classifier", "Number of Trees")
 


# In[271]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ =         learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Learning curve plot
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


fig, axes = plt.subplots(3, 3, figsize=(10, 15))

# Comparison of Naive Bayes, RF and KNN algo 
clf_gauss = GaussianNB()
plot_learning_curve(clf_gauss, "(Naive Bayes)", X, y, axes=axes[:, 0], ylim=(0.5, 1.01),
                    cv=cv, n_jobs=4)

clf_rf = RandomForestClassifier(n_estimators = best_estimator)
plot_learning_curve(clf_rf, "(Random Forest)", X, y, axes=axes[:, 1], ylim=(0.5, 1.01),
                    cv=cv, n_jobs=4)

clf_knn = KNeighborsClassifier(n_neighbors = best_neighbor)
plot_learning_curve(clf_knn, "(KNN Classifier)", X, y, axes=axes[:, 2], ylim=(0.5, 1.01),
                    cv=cv, n_jobs=4)

plt.show()

# Naive Bayes, RF and KNN have similar performance. Naives Bayes and KNN algorithms scale better for enormous data. 


# In[272]:


def cal_tn(Y_test, y_pred): return confusion_matrix(Y_test, y_pred)[0,0]
def cal_fp(Y_test, y_pred): return confusion_matrix(Y_test, y_pred)[0,1]
def cal_fn(Y_test, y_pred): return confusion_matrix(Y_test, y_pred)[1,0]
def cal_tp(Y_test, y_pred): return confusion_matrix(Y_test, y_pred)[1,1]

def tpr(Y_test,y_pred): 
    tp = cal_tp(Y_test,y_pred)
    fn = cal_fn(Y_test,y_pred)
    return round((tp / (tp + fn)),2)

def tnr(Y_test,y_pred): 
    tn = cal_tn(Y_test,y_pred)
    fp = cal_fp(Y_test,y_pred)
    return round((tn / (tn + fp)),2)

def fpr(Y_test,y_pred): 
    tn = cal_tn(Y_test,y_pred)
    fp = cal_fp(Y_test,y_pred)
    
    return round((fp / (tn + fp)),2)

def fnr(Y_test,y_pred): 
    tp = cal_tp(Y_test,y_pred)
    fn = cal_fn(Y_test,y_pred)
    return round((fn / (tp + fn)),2)

def Recall(Y_test,y_pred):
    tp = cal_tp(Y_test,y_pred)
    fn = cal_fn(Y_test,y_pred)
    return round((tp / (tp + fn)),2)

def Precision(Y_test,y_pred):
    tp = cal_tp(Y_test,y_pred)
    fp = cal_fp(Y_test,y_pred)
    return round((tp / (tp + fp)),2)

def F1Score(Y_test,y_pred):
    tp = cal_tp(Y_test,y_pred)
    fp = cal_fp(Y_test,y_pred)
    fn = cal_fn(Y_test,y_pred)
    return round(((2*tp) / ((2*tp) + fp+fn)),2)

def Accuracy(Y_test,y_pred):
    tn = cal_tn(Y_test,y_pred)
    tp = cal_tp(Y_test,y_pred)
    fp = cal_fp(Y_test,y_pred)
    fn = cal_fn(Y_test,y_pred)
    return round(((tp + tn) / (tp + fp + fn + tn)),2)

def Error(Y_test,y_pred):
    tn = cal_tn(Y_test,y_pred)
    tp = cal_tp(Y_test,y_pred)
    fp = cal_fp(Y_test,y_pred)
    fn = cal_fn(Y_test,y_pred)
    return round(((fp + fn) / (tp + fp + fn + tn)),2)

def BACC(Y_test,y_pred):
    tn = cal_tn(Y_test,y_pred)
    tp = cal_tp(Y_test,y_pred)
    fp = cal_fp(Y_test,y_pred)
    fn = cal_fn(Y_test,y_pred)
    return round(0.5*((tp / (tp + fn))+(tn / (fp + tn))),2)

def TSS(Y_test,y_pred):
    tn = cal_tn(Y_test,y_pred)
    tp = cal_tp(Y_test,y_pred)
    fp = cal_fp(Y_test,y_pred)
    fn = cal_fn(Y_test,y_pred)
    return round((tp / (tp + fn))-(fp / (fp + tn)),2)

def HSS(Y_test,y_pred):
    tn = cal_tn(Y_test,y_pred)
    tp = cal_tp(Y_test,y_pred)
    fp = cal_fp(Y_test,y_pred)
    fn = cal_fn(Y_test,y_pred)
    return round((2*((tp * tn)-(fp * fn)))/(((tp + fn)*(fn + tn))+((tp + fp)*(fp + tn))),2)

def cal_mean(dict_score):
    df = pd.DataFrame.from_dict(dict_score, orient='index')
    df['mean'] = df.mean(axis=1)
    return df


# In[273]:


from sklearn.metrics import confusion_matrix,make_scorer
scoring = {'tp': make_scorer(tp),'tn': make_scorer(tn),'fp': make_scorer(fp),'fn': make_scorer(fn),'tpr': make_scorer(tpr),
           'tnr':make_scorer(tnr),'fpr':make_scorer(fpr),'fnr':make_scorer(fnr),'recall':make_scorer(Recall),'precision':make_scorer(Precision),'F1Score':make_scorer(F1Score),
           'Accuracy':make_scorer(Accuracy),'Error':make_scorer(Error),'BACC':make_scorer(BACC),'TSS':make_scorer(TSS),
           'HSS':make_scorer(HSS)}


# In[274]:


clf_gauss_score = cross_validate(clf_gauss,X,y,scoring = scoring,cv=cv)
df_gauss = cal_mean(clf_gauss_score)
df_gauss.head(20)


# In[275]:


clf_rf_score = cross_validate(clf_rf,X,y,scoring = scoring,cv=cv)
df_rf = cal_mean(clf_rf_score)
df_rf.head(20)


# In[276]:


clf_knn_score = cross_validate(clf_knn,X,y,scoring = scoring,cv=cv)
df_knn = cal_mean(clf_knn_score)
df_knn.head(20)


# In[277]:



final_result = {'Type of classifier':['KNN', 'RF', 'Gaussian NB'],
        'Test Accuracy Mean':[df_knn.loc["test_Accuracy"]["mean"],df_rf.loc["test_Accuracy"]["mean"],df_gauss.loc["test_Accuracy"]["mean"]]}

df_final_result = pd.DataFrame(final_result)
  
df_final_result.head()


# In[ ]:




