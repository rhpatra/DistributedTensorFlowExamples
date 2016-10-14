"""
This script is a demo for solving a linear model using the same data sets in TensorFlow Linear Model Tutorial 
	https://www.tensorflow.org/versions/r0.11/tutorials/wide/index.html#tensorflow-linear-model-tutorial
"""
import pandas as pd
import tempfile
# for python3 use urllib.request
# for python2 use urllib
import urllib.request
import tensorflow as tf
import os

def download_data():
	data_name = "adult.data"
	test_name = "adult.test"
	if not os.path.exists(data_name) or not os.path.exists(test_name):
		print ("\033[1;31Downloading....\033[0m")
		with open(data_name, 'w') as data_file:
			# Python3 urllib.request.urlretrieve("")
			# Python2 urllib.urlretrieve("")
			urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", filename=data_file.name)

		with open(test_name, 'w') as test_file:
			urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", filename=test_file.name)

	COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

	print ("\033[1;31mReading data from files\033[0m")
	with open(data_name, 'r') as train_file:
		df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)

	with open(test_name, 'r') as test_file:
		df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

	return df_train, df_test

def main():
	df_train, df_test = download_data()

	# Creating label column
	LABEL_COLUMN = "label"
	df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
	df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

	# Classification types.
	CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
	                       "relationship", "race", "gender", "native_country"]
	CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

	######## Construct sparse and dense featuers columns
	education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)

	# The keys here should be "Female/Male" rather than "female/male" which is displayed in the tutorial.
	gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])

	race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=[
	  "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])
	marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
	relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
	workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
	occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
	native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

	age = tf.contrib.layers.real_valued_column("age")
	education_num = tf.contrib.layers.real_valued_column("education_num")
	capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
	capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
	hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

	age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
	education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))
	age_buckets_x_race_x_occupation = tf.contrib.layers.crossed_column([age_buckets, race, occupation], hash_bucket_size=int(1e6))

	#############

	# Create a column 'id' for SVM. The type of elements should be string according to SVM documentation.
	ID_COLUMN = 'id'
	df_train[ID_COLUMN] = df_train.index.astype(str)
	df_test[ID_COLUMN] = df_test.index.astype(str)

	# defining input functions
	def svm_input_fn(df):
	  # Creates a dictionary mapping from each continuous feature column name (k) to
	  # the values of that column stored in a constant Tensor.
	  continuous_cols = {k: tf.constant(df[k].values)
	                     for k in CONTINUOUS_COLUMNS}
	  # Creates a dictionary mapping from each categorical feature column name (k)
	  # to the values of that column stored in a tf.SparseTensor.
	  categorical_cols = {k: tf.SparseTensor(
	      indices=[[i, 0] for i in range(df[k].size)],
	      values=df[k].values,
	      shape=[df[k].size, 1])
	                      for k in CATEGORICAL_COLUMNS}
	  
	  # for idx
	  idx_cols = {ID_COLUMN: tf.constant(df[ID_COLUMN].values, dtype=tf.string)}
	  # Merges the two dictionaries into one.
	  # for python 2
	  # feature_cols = dict(idx_cols.items() + continuous_cols.items() + categorical_cols.items())
	  feature_cols = dict(idx_cols.items() | continuous_cols.items() | categorical_cols.items())
	  # Converts the label column into a constant Tensor.
	  label = tf.constant(df[LABEL_COLUMN].values)
	  # Returns the feature columns and the label.
	  return feature_cols, label

	def svm_train_input_fn():
	  return svm_input_fn(df_train)

	def svm_eval_input_fn():
	  return svm_input_fn(df_test)

	# Creating a svm instance by feeding all the ingradients we build so far.
	svm_estimator = tf.contrib.learn.SVM(example_id_column=ID_COLUMN, feature_columns=[
	  race, age_buckets, education_num, age, capital_loss, gender, capital_gain, hours_per_week], l1_regularization=0.0, l2_regularization=0.0)

	# Feed training datasets.
	print("\033[1;31mBegin training ... \033[0m")
	svm_estimator.fit(input_fn=svm_train_input_fn, steps=200)

	# Evaluate on the test sets
	print("\033[1;31mBegin testing ... \033[0m")
	metrics = svm_estimator.evaluate(input_fn=svm_eval_input_fn, steps=1)

	# Display results
	for key in sorted(metrics):
	    print ("\033[1;31m%s: %s\033[0m" % (key, metrics[key]))

if __name__ == '__main__':
	main()