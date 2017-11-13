#!/usr/bin/env python

#################################
#
#	Titanic machine learning model for Kaggle comp being
#	used a learning vehicle for ML and TensorFlow
#
#	A TensorFlow supervied neural network using binary 
#	classification
#
#	Objective of the comp is to provide back to Kaggle
#	a prediction for each individual in the test file
#	whether they will survive or not. Obviously test 
#	file lacks survived column
#
#	Train file provides the labeled data for training
#	the neural network and this is divided in training
#	and validation 
#
#	Expects to run in python 3 with TensorFlow installed
#
#################################

#%matplotlib inline
import numpy as np
import pandas as pd
import re as re

#	this switches off annoying TF messages about capabilities
#	available in GPU which haven't been compiled in
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

from collections import OrderedDict
from enum import Enum

#################################
#
#	Helper class to hold the metrics
#	calculations components
#
#################################
class metricholder:

	#	name of nmetric for print
	m_metricname =  ""

	#	TF graph update metrics operation returned by tf.metrics.*** call
	m_tf_update_op = None
	
	#	TF graph value calculator operation returned by tf.metrics.*** call
	m_tf_calc_op = None

	# 	holder for value returned above TF function
	m_value = 0.0

	#	Initialize, force declaration of all values
	def __init__(self,_metricname,_tf_calcop,_tf_updateop,_value):

		self.m_metricname = _metricname
		self.m_tf_calc_op = _tf_calcop
		self.m_tf_update_op = _tf_updateop
		self.m_value = _value

# metricholder class def

#################################
#
#	Tags for hyperparameter dict
#
#################################
class HPTag(Enum):

	sig_positive_theshold 	= 1
	num_hidden_layer_1 		= 2
	num_output_layer 		= 3
	logdir 					= 4
	learning_rate 			= 5
	num_steps 				= 6
	display_step 			= 7
	L2_rate 				= 8
	metrics_tag 			= 9
	num_fare_categories 	= 10
	num_age_categories 		= 11
	validation_percentage 	= 12

# 	HPTag class def

#################################
#
#	Main()
#	Load train and test material, cleanup and select features,
#	build and train model, validate and run test material 
#	through leaned model to generate prediction file
#
#################################
def Main(_justgenfile=False):

	print("Start")

	hypers = GenerateHyperParameterDict()

	# 	read files in panda DataFrame structure. The rationale for keeping
	#	training and test together is feature processing done training
	#	more than likely is required in test as it become model dependant
	train = pd.read_csv('./train.csv', header = 0, dtype={'Age': np.float64})
	test  = pd.read_csv('./test.csv' , header = 0, dtype={'Age': np.float64})
	full_data = [train, test]

	#	cleanup up dataset adding derivative features. this is done on both
	# 	the training and test set as there's an expectations theres as 
	# 	structural expectation build into the mdoel 
	RunAnalysisAndClean(hypers,train,full_data)

	#	dump out the processed files for review before we modify some
	#	of the features to be more processable in neural network
	train.to_csv("train_after_analysis.csv")
	test.to_csv("test_after_analysis.csv")

	if( _justgenfile == True ):
		return

	# 	turn specific fields into numerical values as its easier for the 
	# 	train process
	Numericalize(full_data)

	# 	reduce the feature set to those we really want to train the
	# 	model on
	print("\nReshape train & test data by dropping uninteresting columns")
	#drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch','CategoricalAge', 'CategoricalFare', "FamilySize"]
	drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch','CategoricalAge', 'CategoricalFare', "FamilySize"]
	train = train.drop(drop_elements, axis = 1)
	
	# 	save PassengerId list for manufacturing prediction file
	# 	reshape to fit into model of number of examples being along axis 1	
	pidlist = test.values[:,:1].T

	# 	now drop the same columns on the test set
	test  = test.drop(drop_elements, axis = 1)

	#	print out features we're going the be training on
	#	skip Survived as this is part of this set but will
	#	eventually be split out to form the labels (Y)
	cols = list(train.columns.values)
	print("\nFeatures used for training")
	for c in cols:
		if( c != "Survived" ):
			print("  " + c)

	#	get underlying values from DataFrames
	train = train.values

	# 	Reshape the dataset to break out the labels and fit the normal orientation
	# 	expected i.e
	#		X = (features x num of examples)
	#		Y = (number of classifications (1 here) x num of examples
	X = train[0::, 1::].T  	
	Y = train[0::, 0]
	Y = Y.reshape(1,Y.shape[0])

	print("\nX/Y shapes after processing")
	print("  X shape = " + str(X.shape))
	print("  Y shape = " + str(Y.shape))

	#	Run Feature analysis 
	FeatureAnalysis(X,Y,cols[1:])

	tf_session = tf.Session()

	positive_theshold = 0.5

	# 	run training with the training examples and associated labels
	#	returing the model (logits) and the input variable which is
	#	used to feed inputs to the model. The returned model essential
	#	carries the learn pararmeters
	tf_logits, tf_inputs = Train(hypers,tf_session,X,Y)

	#	get test variables and reshape to suit the model. In this
	#	situation  we're putting the test set through the model
	#	and saving the predictions as Kaggle will be providing 
	#	accuracy !
	test  = test.values
	X = test.T

	print("\n-- Test --------------------")

	#	run test set through the trained model and get the
	#	predicted results 
	predictions = Predict(hypers,tf_session,tf_logits,tf_inputs,X)

	tf_session.close()

	#	write predicitons to the file in the expected format 
	#	Kaggel expects
	WritePredictions(predictions,X,pidlist,"TestPredictions.txt")

	print("Finish")

# main()

#################################
#
#	GenerateHyperParameterDict
#	generate static config dictionary for hyperparameters etc
#
#	no return
#
#################################
def GenerateHyperParameterDict():

	hypers = {}

	hypers[HPTag.sig_positive_theshold] 	= 0.5
	hypers[HPTag.num_hidden_layer_1]		= 128
	hypers[HPTag.num_output_layer] 			= 1
	hypers[HPTag.logdir] 					= "."
	hypers[HPTag.learning_rate] 			= 0.001
	hypers[HPTag.num_steps] 				= 200000
	hypers[HPTag.display_step] 				= 1000
	hypers[HPTag.L2_rate] 					= 0.01
	hypers[HPTag.metrics_tag] 				= "metrics_tag"
	hypers[HPTag.num_fare_categories] 		= 5
	hypers[HPTag.num_age_categories] 		= 8 
	hypers[HPTag.validation_percentage] 	= 0.2

	return hypers

#	GenerateHyperParameterDict()

#################################
#
#	RunAnalysisAndClean()
#	Run analysis on set and add new columns 
#
#	no return
#
#################################
def RunAnalysisAndClean(_hypers,_train,_full_data):

	print (_train.info())

	#print(_train['Survived'].groupby([_train['Pclass'],_train['Survived']]).count())

	print("Survivors by class")
	print(_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

	print("\n")
	print("Survivors by sex")
	print(_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())

	print("\n")
	print("Create FamilySize from SibSp & Parch")
	print("Surviors by family size")
	for dataset in _full_data:
		dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
	print (_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

	print("\n")
	print("Create IsAlone based on Family Size = 1")
	print("Surviors by IsAlone")
	for dataset in _full_data:
		dataset['IsAlone'] = 0
		dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
	print(_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

	print("\n")
	print("Fill out NaN with S")
	print("Surviors by embarkation point")
	for dataset in _full_data:
		dataset['Embarked'] = dataset['Embarked'].fillna('S')
	print (_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

	print("\n")
	print("Fill out NaN median fare")
	print("Create CategoricalFare by quartiling population and taking range with pop")
	print("Surviors by CategoricalFare ")
	for dataset in _full_data:
		dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
		dataset['CategoricalFare'] = pd.qcut(dataset['Fare'], _hypers[HPTag.num_fare_categories])
	print(_train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())

	print("\n")
	print("Create Title by parsing out title from name")
	print("Cross tabulate Title breaking down by Sex")
	for dataset in _full_data:
		dataset['Title'] = dataset['Name'].apply(GetTitle)

	print(pd.crosstab(_train['Title'], _train['Sex']))

	print("\n")
	print("Replace rarely used Title values by 'Rare' value, reduces to 5 Title values, and clean up the rest")
	print("Survivors by Title")
	for dataset in _full_data:
		dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
			'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

		dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
		dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
		dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
	print(_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

	np.random.seed(2)

	#	titles and their default average age if there are no existing ages in the same group to average on
	titlelist = {"Mr":32,"Miss":14,"Mrs":34,"Master":4,"Rare":43}
	for dataset in _full_data:

		for title, defaultage in titlelist.items():

			for alone in range(0,2):

				age_null_count = len(dataset.loc[(dataset.Title == title) & (dataset.IsAlone == alone) & (dataset.Age.isnull() == True),'Age'])
				if age_null_count == 0:
					continue

				age_avg = dataset.loc[(dataset.Title == title) & (dataset.IsAlone == alone) & (dataset.Age.isnull() == False),'Age'].mean()
				age_std = dataset.loc[(dataset.Title == title) & (dataset.IsAlone == alone) & (dataset.Age.isnull() == False),'Age'].std()

				#	possible if there is only 1 row in the group being processed
				if( np.isnan(age_avg) ):
					age_avg = defaultage
					age_std = 0
					random_age_list = np.array([age_avg])
				else:
					np.random.seed(2)
					random_age_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

				print("Title/isAlone = " + title + "/" + str(alone) + " : Count = " + str(age_null_count) + " Avg = " + str(age_avg) + " Std = " + str(age_std))
				dataset.loc[(dataset.Title == title) & (dataset.IsAlone == alone) & (dataset.Age.isnull() == True),'Age'] = random_age_list

		dataset['Age'] = dataset['Age'].astype(int)
		dataset['CategoricalAge'] = pd.cut(dataset['Age'], _hypers[HPTag.num_age_categories])

	print(_train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())

	return

# RunAnalysisAndClean()

#################################
#
#	GetTitle()
#	Extracts the title element from the string in _name
#
#	return title string
#
#################################
def GetTitle(_name):

	title_search = re.search(' ([A-Za-z]+)\.', _name)

	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

# 	GetTitle()

#################################
#
#	Numericalize()
#	Map features to be used into easier to consume numerical representation
#
#	no return
#
#################################
def Numericalize(_full_data):

	for dataset in _full_data:

		# Mapping Sex 0 or 1
		dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

		# Mapping title string into 1 to 5
		title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
		dataset['Title'] = dataset['Title'].map(title_mapping)
		dataset['Title'] = dataset['Title'].fillna(0)

		# Mapping Embarked into 1 to 3
		dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

		# 	Auto map Fare to category number based on number of CategoricalFares produced in RunAnalysisAndClean
		for dataset in _full_data:

			catbinslist = dataset[['CategoricalFare', 'Fare']].groupby(['CategoricalFare'], as_index=False).mean()
			for i in range(0,len(catbinslist)):
				dataset.loc[(dataset.CategoricalFare == (catbinslist['CategoricalFare'][i])),'Fare'] = i

		#	Auto map age to category number based on number of CategoricalAges produced in RunAnalysisAndClean
		for dataset in _full_data:

			catbinslist = dataset[['CategoricalAge', 'Age']].groupby(['CategoricalAge'], as_index=False).mean()
			for i in range(0,len(catbinslist)):
				dataset.loc[(dataset.CategoricalAge == (catbinslist['CategoricalAge'][i])),'Age'] = i


	return

#	Numericalize()

#################################
#
#	FeatureAnalysis()
#	Runs feature choice through a couple of statistic reviews to
#	establish  
#
#	no return	
#
#################################
def FeatureAnalysis(_X,_Y,_cols):

	print("\nAnalysis across selected features")
	#	SelectKBest expects 
	#		X 	[num samples x features]
	#		Y 	[num samples]
	X = _X.T
	Y = _Y.T

	test = SelectKBest(score_func=chi2, k="all")
	fit = test.fit(X, Y)

	# summarize scores
	np.set_printoptions(precision=3)
	
	scoredict = {}
	for i in range(0,len(_cols)):
		scoredict[_cols[i]] = fit.scores_[i]
	scoredict = OrderedDict(sorted(scoredict.items(), key=lambda t: t[1], reverse=True))

	print(" SelectKBest scores")
	for key, value in scoredict.items():
		print("   " + '{:<11}'.format(key) + " = " + str(value))

	Y = np.ravel(Y)
	model = ExtraTreesClassifier()
	model.fit(X, Y)

	scoredict = {}
	for i in range(0,len(_cols)):
		scoredict[_cols[i]] = model.feature_importances_[i]
	scoredict = OrderedDict(sorted(scoredict.items(), key=lambda t: t[1], reverse=True))

	print("\n ExtraTreesClassifier scores")
	for key, value in scoredict.items():
		print("   " + '{:<11}'.format(key) + " = " + str(value))

	variance_pct = .99
	 
	# Create PCA object
	pca = PCA(n_components=variance_pct)
	 
	# Transform the initial features
	X_transformed = pca.fit_transform(X,Y)
	 
	# Create a data frame from the PCA'd data
	pcaDataFrame = pd.DataFrame(X_transformed)
	 
	print(str(pcaDataFrame.shape[1]) + " components describe " + str(variance_pct)[1:] + " % of the variance")

	return

#	FeatureAnalysis()

#################################
#
#	BuildModel()
#	Creates TF graph for basic 1 hidden layer network
#		_num_input_features		number of feature brough in be X essential size of axis 0 on X
#		Layer 1 # nodes			assuming we're only building 1 hidden layer, number of nodes
#		Output # nodes 			number of output nodes, linked to train and eval current really fixed to 1
#
#	Adding in L2 regularization so we need to manufacture the matrices to hold the adjustments 
#	sized on the W matrices. These are placed in LOCAL_VARIABLES and will be retrieved and used
#	when we define the cost function as that where L2 is applied
#
#	returns	TF nodes for logits, inputs, labels
#
#################################
def BuildModel(_hypers,_num_input_features):

	num_hidden_layer_1 = _hypers[HPTag.num_hidden_layer_1]
	num_output_layer = _hypers[HPTag.num_output_layer]

	#	take the run to run variablity out of the random intialization
	#	values
	fixseed = 2

	#	these are the TF values for the feed_dict, basically input and labels
	tf_X = tf.placeholder(tf.float32,name="X")
	tf_Y = tf.placeholder(tf.float32,name="Y")

	#	Layer 1 variables 
	tf_W_1 = tf.Variable(tf.random_normal([num_hidden_layer_1,_num_input_features],seed=fixseed), name="W_1")
	tf_B_1 = tf.Variable(tf.zeros([num_hidden_layer_1,1]), name="B_1")
	tf_L2_1	= tf.Variable(tf.random_normal([num_hidden_layer_1,_num_input_features], seed=fixseed), name="L2_1", collections=[tf.GraphKeys.LOCAL_VARIABLES])

	#	Output layer
	tf_W_Out = tf.Variable(tf.random_normal([num_output_layer,num_hidden_layer_1], seed=fixseed), name="W_Out")
	tf_B_Out = tf.Variable(tf.zeros([num_output_layer,1]), name="B_Out")
	tf_L2_Out = tf.Variable(tf.random_normal([num_output_layer,num_hidden_layer_1], seed=fixseed), name="L2_Out", collections=[tf.GraphKeys.LOCAL_VARIABLES])

	#print("tf_X     = " + str(tf_X))
	#print("tf_Y     = " + str(tf_Y))
	#print("tf_W_1   = " + str(tf_W_1))
	#print("tf_B_1   = " + str(tf_B_1))
	#print("tf_W_Out = " + str(tf_W_Out))
	#print("tf_B_Out = " + str(tf_B_Out))

	#	create input to layer 1 using relu activation
	tf_A_1 = tf.nn.relu(tf.add(tf.matmul(tf_W_1,tf_X), tf_B_1))

	#	create layer 1 to output layer. activation function supplied 
	#	when cost function defined
	tf_Z_Out = tf.add(tf.matmul(tf_W_Out,tf_A_1), tf_B_Out, name="Logits")

	return( tf_Z_Out, tf_X, tf_Y )

#	BuildModel()

#################################
#
#	GetTFVarFromLocal()
#	Obtains a TF variable from LOCAL_VARIABLES using name
#
#	returns TF var
#
#################################
def GetTFVarFromLocal(_name):

	vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
	for v in vars:
		
		#	op.name drops the :nn reference on the back of the name
		if v.op.name == _name:
			return( v )

	return( None )

#	GetTFVarFromLocal()

#################################
#
#	TrainModel()
#	Sets up TF training graph using the TF node retuned by the BuildModel 
#	call (logits) and a sigmoid activation function. As we're using L2 
#	regularization. Executing the node with X & Y will end up with the learning
#	parameters in W & B for each layer
#
#	returns TF node for training
#
#################################
def TrainModel(_hypers,_tf_logits,_tf_labels):

	# obtain the L2 Regularization variables for local store
	tf_L2_1 = GetTFVarFromLocal("L2_1")
	tf_L2_Out = GetTFVarFromLocal("L2_Out")

	#	set cost function
	tf_cost_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=_tf_logits,labels=_tf_labels))

	#	if the L2 variables exist then add the L2 regularization adjustment for both layers
	if( tf_L2_1 and tf_L2_Out ):
		tf_cost_op = tf_cost_op + (_hypers[HPTag.L2_rate] * tf.nn.l2_loss(tf_L2_1)) + (_hypers[HPTag.L2_rate] * tf.nn.l2_loss(tf_L2_Out))

	#	create optimizer 
	#tf_optimizer = tf.train.AdamOptimizer(learning_rate=_hypers[HPTag.learning_rate])
	tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate=_hypers[HPTag.learning_rate])
	tf_train_op = tf_optimizer.minimize(tf_cost_op)

	return( tf_train_op, tf_cost_op )

#	TrainModel()

#################################
#
#	PredictionCalcModel()
#	TF component which provides prediction output from logits 
#	used in both training and prediction. 
#	Executing this with model node and feed dict with provide
#	prediction values which can be evaluated verse labels or
#	written out
#
#	returns TF node representing prediction calc
#
#################################
def PredictionCalcModel(_tf_logits,_positive_theshold):

	#	given we're using 
	tf_prediction = tf.greater(tf.sigmoid(_tf_logits),_positive_theshold)

	return( tf_prediction )

#	PredictionCalcModel()

#################################
#
#	EvaluateModel
#		Sets up a number of calculations which will be performed on the
#		prediction values produced by the model. Each TensorFlow metric
#		created here creates its own variables which are allocated to 
#		the tf.GraphKeys.LOCAL_VARIABLES collection and held under the 
#		metrictag. In ReInitializeAndUpdateMetrics() we intailize the
#		variables and execute the graphs nodes. 
#		The list is important as these calcs need to be run in order
#		as there is a dependency e.g F1
#
#	returns list of metric calculations
#
#################################
def EvaluateModel(_hypers,_tf_logits,_tf_labels):

	tf_prediction = PredictionCalcModel(_tf_logits, _hypers[HPTag.sig_positive_theshold])

	metricslist = list()
	metrictag = _hypers[HPTag.metrics_tag]

	tf_value, tf_update_op = tf.metrics.accuracy(_tf_labels, tf_prediction, name=metrictag)
	metricslist.append(metricholder("accuracy",tf_value,tf_update_op,0.0))

	tf_value, tf_update_op = tf.metrics.precision(_tf_labels, tf_prediction, name=metrictag)
	metricslist.append(metricholder("precision",tf_value,tf_update_op,0.0))
	tf_precision = tf_value

	tf_value, tf_update_op = tf.metrics.recall(_tf_labels, tf_prediction, name=metrictag)
	metricslist.append(metricholder("recall",tf_value,tf_update_op,0.0))
	tf_recall = tf_value

	tf_value, tf_update_op = tf.metrics.auc(_tf_labels, tf_prediction, name=metrictag)
	metricslist.append(metricholder("AUC",tf_value,tf_update_op,0.0))

	tf_f1 = (2 * (tf_recall * tf_precision)) / (tf_recall + tf_precision)
	metricslist.append(metricholder("F1",tf_f1,None,0.0))

	return( metricslist )

#	EvaluateModel()

#################################
#
#	IntializeUpdateAndCalcMetrics
#		Really a convinence function here as we can run all metric related
#		function together. If we were using mini batchs then 
#
#	returns None
#
#################################
def IntializeUpdateAndCalcMetrics(_hypers,_tf_session,_metricslist,_feed_dict):

	InitializeMetrics(_hypers,_tf_session)

	UpdateMetrics(_tf_session,_metricslist,_feed_dict)

	CalculateMetrics(_tf_session,_metricslist,_feed_dict)

#	IntializeUpdateAndCalcMetrics()

#################################
#
#	InitializeMetrics
#		Resets the local variables used for metrics. The expectation is all 
#		metrics variables were created under the same _metrictag and placed
#		in the LOCAL_VARIABLES collection. see EvaluateModel()
#
#	returns None
#
#################################
def InitializeMetrics(_hypers,_tf_session):

	running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=_hypers[HPTag.metrics_tag])
	running_vars_initializer = tf.variables_initializer(var_list=running_vars)

	_tf_session.run(running_vars_initializer)

	return

#	InitializeMetrics()

#################################
#
#	UpdateMetrics
#		Iterates through the metrics list and call to update the internal
#		variables for each metric where there is an update function. 
#
#	returns None
#
#################################
def UpdateMetrics(_tf_session,_metricslist,_feed_dict):

	for mholder in _metricslist:

		if( mholder.m_tf_update_op is not None ):
			_tf_session.run(mholder.m_tf_update_op,feed_dict=_feed_dict)

	return

#	UpdateMetrics()

#################################
#
#	CalculateMetrics
#		Iterates through the metrics list and call to produce the calculated
#		variables for each metric where there is an calculate function. 
#
#	returns None
#
#################################
def CalculateMetrics(_tf_session,_metricslist,_feed_dict):

	for mholder in _metricslist:

		if( mholder.m_tf_calc_op is not None ):
			mholder.m_value = _tf_session.run(mholder.m_tf_calc_op,feed_dict=_feed_dict)

	return

#	CalculateMetrics()

#################################
#
#	PrintMetrics
#
#	returns None
#
#################################
def PrintMetrics(_pretag,_metricslist):

	print(_pretag)

	for mholder in _metricslist:

		print(" " + '{:<10}'.format(mholder.m_metricname) + " = " + str(mholder.m_value))

	return

# PrintMetrics()

#################################
#
#	Train
#		Creates model, runs train and validation set through and
#		returns the fitted model
#
#	return logits
#
#################################
def Train(_hypers,_tf_session,_X,_Y):

	print("\n-- Training ----------------")

	#	Divide up the input set into train and validation	
	num_validationset = (int)(_X.shape[1] * (1 - _hypers[HPTag.validation_percentage]))
	num_trainingset = _X.shape[1] - num_validationset

	X_train = _X[::, 0:num_trainingset]
	Y_train = _Y[::, 0:num_trainingset]

	X_validation = _X[::, num_trainingset::]
	Y_validation = _Y[::, num_trainingset::]

	print("X train = " + str(X_train.shape))
	print("Y train = " + str(Y_train.shape))

	print("X validation = " + str(X_validation.shape))
	print("Y validation = " + str(Y_validation.shape))

	#	hyper parameters
	num_input_features = _X.shape[0]

	#	create TF nodes for the neural network
	tf_logits, tf_inputs, tf_labels = BuildModel(_hypers,num_input_features)

	#	create TF nodes for training the network
	tf_train_op, tf_cost_op = TrainModel(_hypers,tf_logits,tf_labels)

	#	set up TensorBoard writer
	tf_saver = tf.train.Saver()
	tf.summary.scalar('loss', tf_cost_op)
	tf_summary = tf.summary.merge_all()
	tf_summary_writer = tf.summary.FileWriter(_hypers[HPTag.logdir], _tf_session.graph)

	_tf_session.run(tf.global_variables_initializer())
	_tf_session.run(tf.local_variables_initializer())

	checkpoint_file = os.path.join('model.ckpt')
	tf_saver.save(_tf_session,checkpoint_file)

	print("Run training")

	feed_dict = {tf_inputs : X_train, tf_labels : Y_train}

	for step in range(1,_hypers[HPTag.num_steps] + 1):

		opt, loss = _tf_session.run([tf_train_op, tf_cost_op], feed_dict=feed_dict)

		if( step % _hypers[HPTag.display_step] == 0 or step == 1):
			print("   Step = " + str(step) + " Loss = " + str(loss))

			tf_summary_str = _tf_session.run(tf_summary, feed_dict=feed_dict)
			tf_summary_writer.add_summary(tf_summary_str, step)
			tf_summary_writer.flush()

	print("Final Train Loss= " + str(loss))
	
	metricslist = EvaluateModel(_hypers,tf_logits,tf_labels)

	IntializeUpdateAndCalcMetrics(_hypers,_tf_session,metricslist,feed_dict)
	PrintMetrics("Training metrics",metricslist)

	tf_saver.save(_tf_session,checkpoint_file)

	print("\n-- Validation --------------")

	feed_dict = {tf_inputs : X_validation, tf_labels : Y_validation}

	IntializeUpdateAndCalcMetrics(_hypers,_tf_session,metricslist,feed_dict)
	PrintMetrics("Validation metrics",metricslist)

	#print("y_hat = " + str(tf_prediction.eval(session=tf_sess,feed_dict=feed_dict)))
	#print("y     = " + str(labels.shape))

	return( tf_logits, tf_inputs )

# 	Train

#################################
#
#	Predict
#		Run examples (expected to be minus labels) through existing model to
#		produce predictions
#
#################################
def Predict(_hypers,_tf_session,_tf_logits,_tf_inputs,_X):

	print("X test shape = " + str(_X.shape))

	tf_prediction = PredictionCalcModel(_tf_logits,_hypers[HPTag.sig_positive_theshold])

	feed_dict = {_tf_inputs : _X}

	predictions = _tf_session.run(tf_prediction,feed_dict=feed_dict)

	return( predictions )

#	Predict()

#################################
#
#	WritePredictions
#		Write predicitions to file using the expect format
#			PassengerId,Survived
#			892,0
#			893,1
#			Etc.
#
#################################
def WritePredictions(_predictions,_X,_pidlist,_fname):

	print("predictions.shape = " + str(_predictions.shape) )
	print("passengers.shape  = " + str(_pidlist.shape) )

	try:

		f = open(_fname,"w")
		f.writelines("PassengerId,Survived")

		for i in range(0,_predictions.shape[1]):
			f.writelines("\n" + str(_pidlist[0,i]) + "," + str(int(_predictions[0,i])))

		f.close()

	except IOError:
		print("Failed to open " + _fname)

	print("Wrote prediction file for test set")
		
#	WritePredictions()

#################################
#
#	Main
#
#################################
if __name__ == "__main__":

	#Main(_justgenfile=True)
	Main()


