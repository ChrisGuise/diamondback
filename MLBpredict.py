# -*- coding: utf-8 -*-

"""
Date: 25 Apr 2018
Author: Chris Guise / Arizona State University
Python for Engineers Final Project - Spring 2018

*******************************************************************************
*                                                                             *
*                             MLBpredict.py                                   *
*        1) Import the full "teamstats.csv" file                              *
*        2) Identify columns to use for analysis                              *
*        3) Use 50/50 train/test split                                        *
*        4) Calculate PCA components / vectors / eigenvalues                  *
*        5) Run ML algorithm + logistic regression on csv data                *
*           to predict playoff appearances (P>0.5)                            *
*           & probability of playoff appearances (P)                          *
*        6) Output PCA components (pcaVectors.csv)                            *
*        7) Report prediction accuracy (predictions.csv)                      *
*                                                                             *
*******************************************************************************

Out of all algorithms, by a process of trial and error found
that logistic regression and random forest had the highest accuracy 
based on random train/test 50/50 splits.
Used Logistic Regression in order to provide continuous probabilities.
    
"""

#need numpy functions
import numpy as np
from io import StringIO

#need pandas for file reading
import pandas as pd

#For plots
import matplotlib.pyplot as plt
import itertools
from matplotlib.backends.backend_pdf import PdfPages

#Found this on StackOverflow
#Use this with matPlotLib to allow for non-overlapping labels
#https://github.com/Phlya/adjustText
#NOTE: NEED TO HAVE adjustText in  PROJECT DIRECTORY,
#      OTHERWISE PLOTS WILL NOT LOAD CORRECTLY! =)
from adjustText import adjust_text

#Use this later to evaluate multiple iterations of PCA vectors
dataframe_collection = {} 

#number of iterations
#Average 100 trials with unique train/test splits to report accuracy %
#Output files are from last trial run.
n_iter = 100
accuracy_sum = 0
avg_accuracy = 0

#Found this on StackOverflow
#Use this with matPlotLib to allow for non-overlapping labels
#https://github.com/Phlya/adjustText
#https://github.com/Phlya/adjustText/blob/master/examples/Examples.ipynb 
def plot_mtcars(adjust=False, *args, **kwargs):
    plt.figure(figsize=(15, 7.5))
    plt.scatter(data[:, 0], data[:, 2], s=55, c='r', edgecolors=(1,1,1,0))
    texts = []
    for x, y, s in zip(data[:, 0], data[:, 2], data[:, 1]):
        texts.append(plt.text(x, y, s, size=14))
    if adjust:
        plt.title('%s iterations' % adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5), **kwargs))

#Done with imports / helper functions
#---------------------------------------------------------------------------------------------------------------------
    

#-----------------------------------------#
# 1) Import the full "teamstats.csv" file #
#-----------------------------------------#
df= pd.read_csv('./teamStats_example.csv', encoding='latin-1',index_col="teamID" )

#Return the column headers names
data_columns = list(df)

#-----------------------------------------#
# 2) Identify columns to use for analysis #
# Set the X and Y lists for later on.     #
#-----------------------------------------#
length = len(data_columns)
data_columnsX = data_columns[0:length-3] 
data_columnsY = data_columns[length-3]    #playoffs

#Note that with the sample dataset, the algorithm is pretty accurate for...
#Finding who won't win the AL or World Series. 
#Need different data points to predict who wins, in the playoffs. 
#data_columnsY = data_columns[length-2]   #NL/AL Champs
#data_columnsY = data_columns[length-1]   #World Series

data_columnsY2 = data_columnsY + 'actual'
data_columnsY3 = data_columnsY + 'prob'

#Found this on StackOverflow. 
#Axis=1 => Columns
#Create a dataframe object called numdf (numbers only)
#Drop all the data columns (it is empty now)
#Join each column after passing through to_numeric (question marks become NaN)
numdf = (df.drop(data_columns, axis=1)
         .join(df[data_columns].apply(pd.to_numeric, errors='coerce')))

#NaN is recognized as a null value. Delete any row where ALL the columns are not NaN.
numdf = numdf[numdf[data_columns].notnull().all(axis=1)]

#Use all input columns (features)
X = numdf[data_columnsX]

#Load the answer key
Y = numdf[data_columnsY]

#-------------------------------#
# 3) Use 50/50 train/test split #
#-------------------------------#       
#Not using cross validation...used this just to split train and test datasets
from sklearn.cross_validation import train_test_split

#Run regression n_iter times
#This reduces the effect of over-fitting for %accuracy calculation purposes
for n in range(0, n_iter):
    
    #train_test split returns XY pairs for train and test datasets.
    #Test size: 50% training data, 50% test data. random_state is a seed for split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.5)
    teamnames = X_test.index.values.tolist()
    
    #Scaling to mean=0, sigma=1
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    
    #sc.fit estimates mean & sigma from the training data
    sc.fit(X_train)
    
    #Standardize the mean and sigma based on the training data to mean=0, sigma=1
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

#-----------------------------------------------------#
# 4) Calculate PCA components / vectors / eigenvalues #
#-----------------------------------------------------#
    #Note that each of the PCA components has multiple correlated features.
    from sklearn.decomposition import PCA
    
    #3 PCA vectors. Each vectors has all the features.
    n_components_pca = 3
    pca = PCA(n_components = n_components_pca)
    
    #PCA transform on train, test datasets
    X_train_std_pca = pca.fit_transform(X_train_std)
    X_test_std_pca = pca.transform(X_test_std)
    
    componentsList = ["PC-1"] #1st element
    for i in range(1,n_components_pca):
        
        componentsList.append("PC-" + str(i+1))      
    
    #Create containers
    df_pcaComp = pd.DataFrame(pca.components_,columns=data_columnsX,index = componentsList)
    dataframe_collection[n] = pd.DataFrame(pca.components_,columns=data_columnsX,index = componentsList)   
    
    #Accuracy score helper function
    from sklearn.metrics import accuracy_score
    
#-------------------------------------------------------#  
# 5) Run ML algorithm + logistic regression on csv data # 
#-------------------------------------------------------#
    #1) import method
    #2) create an instance of the method
    #3) use .fit to train the instance (set model parameters for instance)
    #4) use .predict to test the instance (is the instance predicting output correctly?)
    
    
    ##############################################################
    #LogisticRegression: in 2d space, classify points by a dividing sigmoid shape
    from sklearn.linear_model import LogisticRegression
    
    #LR parameters:
    #LR implements a closed loop feedback for minimization/convergence.
    #LR updates weights after each guess (correct or incorrect)
    #Weights are for logistic sigmoid function (perceptron is for step function)
    #C = regularization constant. Lower C => 
    #Lowering C (regularization) reduces weights, which irons out extreme 
    #parameter values. This may prevent over-fitting to training dataset.
    #Lower C => lower variance/sensitivity.
    #random_state: Seed for random number generator (order of data points to form guess)
  
    #Instance of LR
    lr = LogisticRegression(C=1.0, random_state=0)
    #Train instance (train data)
    lr.fit(X_train_std_pca, Y_train)
    #Predict outcome (test data)
    Y_pred2 = lr.predict(X_test_std_pca)
    #Predict Probability (test data)
    Y_pred2_prob = lr.predict_proba(X_test_std_pca)
    
    #Side by side: algorithm prediction vs. actual result (test data)
    d2 = {'team': teamnames, data_columnsY2: Y_test, data_columnsY: Y_pred2}
    df2 = pd.DataFrame(d2)
    
    #Side by side: algorithm probability vs. actual result (test data)
    d2_prob = {'team': teamnames, data_columnsY2: Y_test, data_columnsY: Y_pred2_prob[:,1]}
    df2_prob = pd.DataFrame(d2_prob)
    
    #Side by side: algorithm probability vs. actual result (test data)
    d2_both = {'team': teamnames, data_columnsY: Y_pred2, data_columnsY3: Y_pred2_prob[:,1], data_columnsY2: Y_test}
    df2_both = pd.DataFrame(d2_both)

    #For accuracy over n iterations
    accuracy_sum = accuracy_sum + accuracy_score(Y_test, Y_pred2)

#Calculate + return accuracy over n_iter (final result)
avg_accuracy = accuracy_sum / n_iter
print('**Average accuracy:'+str(avg_accuracy))

'''
Compute the mean over multiple iterations even when the sign of the PCA
eigenvectors is changing

#df_pcaComp2 = pd.DataFrame(pca.components_,columns=data_columnsX,index = componentsList)

for n in range(0, n_iter-1):
    
    #dataframe_collection[n] = pd.DataFrame(pca.components_,columns=data_columnsX,index = componentsList)
    df_concat = pd.concat((df_pcaComp2, dataframe_collection[n]))
    by_row_index = df_concat.groupby(df_concat.index)
    df_medians = by_row_index.mean()
'''

#-------------------------------------------#
# 6) Output PCA components (pcaVectors.csv) #    
#-------------------------------------------#    
df_pcaComp_transposed = df_pcaComp.transpose()
df_pcaComp_transposed.to_csv('pcaVectors.csv')

#Plot PCA Vectors:

for i in range(1,n_components_pca+1):
    
    #Define a figure, give it a name
    plt.figure(figsize=(15,5))
    PCx = 'PC-' + str(i)
    
    #Sort Ascending
    df_pcaComp_transposed_sorted = df_pcaComp_transposed.loc[:,PCx].sort_values(ascending = False)
    
    #Define the x axis as the index of the sorted features within the PCA eigenvector
    numPoints = len(df_pcaComp_transposed_sorted.index.values)
    indexForPlot = np.arange(1,numPoints+1,1)
    
    #X = index, Y = Weight of feature within eigenvector, Z = labels
    xvals = list(indexForPlot)
    yvals = list(df_pcaComp_transposed_sorted[:])
    zvals = list(df_pcaComp_transposed_sorted.index.values)
    
    #Make it so TotalPct always is positive direction if it matters (<-0.03)
    for i in range (1, len(yvals)):
        if (zvals[i] == 'totalPct'):
            #invert values if totalPct is negative
            if(yvals[i] < -0.03 ):
                    for i in range(0, len(yvals)):
                        yvals[i] = yvals[i] * -1
                    
                
    #Labels are important here...otherwise X axis is just dots. 
    #Makes it much easier to read than having features as the X axis
    labels = zvals
    
    data = pd.DataFrame({'Index' : xvals, 
                         'weight': yvals,
                         'name':   zvals
                         })
    
    data = data.values

    #Found this on StackOverflow
    #use AdjustText so the plot is readable by humans
    plot_mtcars(adjust=True, force_text=(0.2, 1))
    plt.xlabel('Index#', fontsize=16)
    plt.ylabel('Weight', fontsize=16)
    plt.title(PCx, fontsize=16)
    plt.grid(True)

    #Get Current Figure...before plt.show() clears it
    fig1 = plt.gcf()  
    
    #Creates a new figure
    plt.show()
    
    #Save figure as PNG
    fig1.savefig(PCx +'.png', bbox_inches='tight')
    
#-------------------------------------------------#
# 7) Output prediction accuracy (predictions.csv) #    
#-------------------------------------------------#
df2_both_sorted = df2_both.sort_values(by='playoffsprob', ascending = False)
#df2_both_sorted = df2_both.sort_values(by='lg_winprob', ascending = False)
#df2_both_sorted = df2_both.sort_values(by='ws_winprob', ascending = False)

df2_both_sorted.to_csv('predictions.csv')




