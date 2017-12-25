
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings("ignore")


train_list = ['03','03-04','03-05','03-06','03-07','03-08','03-09','03-10','03-11', '03-12','03-13'] #List that iterates over training files
test_list = ['04-05','05-06','06-07','07-08','08-09','09-10','10-11','11-12','12-13', '13-14','14-15'] #List that iterates over test files
classifier = ['KNN','RF','SVM','SGD','LR','NN'] #List for choosing classifiers

#Load the data
train_path = '../Data/numpyFiles/Train/'
test_path =  '../Data/numpyFiles/Test/'

for i in range(0,len(test_list)): #Iterate over seasons

	accuracy = [] #SelectFromModel accuracy 
	pca_accuracy = [] #PCA dimensionality reduction accuracy
	print "For season",test_list[i]

	for clf in classifier: #Iterate over classifiers
		#Set Train and Test features + labels
		train_data = np.load(train_path + 'train'+str(train_list[i])+'.npy')
		test_data = np.load(test_path + 'testStats'+str(test_list[i])+'.npy')
		train_x_old = train_data[:, 0:train_data.shape[1]-1]
		train_y = train_data[:, -1]
		test_x = test_data[:, 0:test_data.shape[1]-1]
		test_y = test_data[:, -1]
		
		#Now check for classifier and declare model accordingly
		if clf == 'KNN':
		    my_model = KNeighborsClassifier() #random_state = 42
		    param_grid = {'n_neighbors':range(1,5), 'weights': ['uniform','distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'metric' : ['euclidean','manhattan','chebyshev','minkowski']}

		if clf == 'RF':
		    my_model = RandomForestClassifier(random_state = 42)
		    param_grid = {'n_estimators' : range(10,230,10), 'criterion' : ['gini','entropy'], 'max_features' : ['auto', 'sqrt', 'log2']}

		if clf == 'SVM':
		    my_model = svm.SVC(random_state = 42)
		    param_grid = {'kernel':['linear','rbf'], 'gamma': [1e-1, 1, 1e1],'C':[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]}
		    	
		if clf == 'SGD':
		    my_model = SGDClassifier(random_state = 42)
		    param_grid={'loss':['log','hinge','modified_huber'],'penalty':['l1','l2'],'alpha':[0.0001,0.001,0.01,0.1,1.0],'n_jobs':[-1,1,2,3]}

		if clf == 'LR':
                    my_model = LogisticRegression(random_state = 42)
		    param_grid={'penalty': ['l2'], 'dual':[False], 'C':[0.1, 0.05, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], 'fit_intercept' :[True, False], 'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag']}

		if clf == 'NN':
		    my_model = MLPClassifier(random_state = 42)
		    param_grid = {'activation':['identity','logistic','tanh','relu'], 'solver': ['lbfgs', 'sgd', 'adam'],'alpha':[0.000001, 0.00001, 0.0001, 0.001, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0], 'learning_rate':['constant', 'invscaling', 'adaptive']}

		#---------------------SELECTFROMMODEL----------------------------------

		#Using LassoCV() as estimator in Feature Selection method SelectFromModel()
		sfm=SelectFromModel(LassoCV())

		#Fitting and transforming the training data
		train_x_new_SFM=sfm.fit_transform(train_x_old,train_y)

		#Transforming the testing data
		test_x_new_SFM=sfm.transform(test_x)

		#The following portion has been commented out for brevity of the terminal output, but we used this to calculate the most important features

		#Printing the features that were selected by the method
		#print("Features selected by feature selection method:") 
		# print (sfm.get_support())
		# li = list(sfm.get_support())
		# print li
		# feats = []
		# for itera in range(len(li)):
		# 	if li[itera]==True:
		# 		feats.append(itera)
		# print feats

		#hyperparameter optimization, done using GridSearchCV
		optimal_model = GridSearchCV(my_model,param_grid)
		optimal_model.fit(train_x_new_SFM, train_y)
		print optimal_model.best_params_
		predictions = []
		predictions = optimal_model.predict(test_x_new_SFM)
		class_counter = 0
		for index_sfm in range(0,len(predictions)):
		    if predictions[index_sfm] == test_y[index_sfm]: #Check if points class matches, update counter if it does
		    	class_counter += 1
		accuracy.append(float(class_counter)/(len(predictions)))
		print "SelectFromModel Accuracy: "+str(accuracy)

	#-------------------------PCA----------------------------------
		
		pca = PCA(n_components=25)
		#Fitting and transforming the training data

		train_x_pca = pca.fit_transform(train_x_old,train_y)
		
		#Transforming the testing data
		test_x_pca = pca.transform(test_x)

		optimal_model.fit(train_x_pca, train_y)
		pca_predictions = []
		pca_predictions = optimal_model.predict(test_x_pca)
		class_counter = 0
		for index_pca in range(0,len(pca_predictions)):
		    if pca_predictions[index_pca] == test_y[index_pca]: #Check if points class matches, update counter if it does
		    	class_counter += 1
		pca_accuracy.append(float(class_counter)/(len(pca_predictions)))
		print "PCA Accuracy: "+str(pca_accuracy)

	# Plotting combined bar charts for SelectFromModel and PCA
	pos=np.arange(6)
	n_groups = 6 #For 6 classifiers
	plt.figure(1, figsize=(10,8))  #10x8 is the aspect ratio for the plot
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	rects1 = plt.bar(index, accuracy, bar_width,
		         alpha=opacity,
		         color='b',
		         label='SelectFromModel')

	rects2 = plt.bar(index + bar_width, pca_accuracy, bar_width,
		         alpha=opacity,
		         color='r',
		         label='PCA Dimensionality Reduction')


	plt.xlabel('Model')
	plt.ylabel('Accuracy')
	plt.xlim(-0.5,7.0) #set x axis range
	plt.ylim(0,0.7) #Set yaxis range
	plt.title('SelectFromModel vs PCA Dimensionality Reduction accuracies')
	plt.xticks(index + bar_width / 2, ('KNN', 'RF', 'SVM', 'SGD', 'LR','NN'))
	plt.legend()

	plt.tight_layout()
	plt.savefig("../Figures/"+str(test_list[i])+".jpg") #Save figure
	plt.clf() #Clear


