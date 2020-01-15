from jikanpy import Jikan
import pandas as pd
import numpy as np
import surprise
import time


# Recommender handles the machine learning to recommend people new anime to watch.
# The data used for the machine learning was scraped from https://myanimelist.net/

class Recommender():
	def __init__(self, df, user_id, testset, algo = 'svd'):
		self.df = df
		self.user_id = user_id
		self.testset = testset
		if algo == 'svd':
			self.algo = surprise.SVD()

	# Use scikit-surprise library to perform matrix factorization
	def surprise_fit(self):
		reader = surprise.Reader(rating_scale=(0,10))
		data = surprise.Dataset.load_from_df(self.df[['userID', 'itemID', 'rating']], reader)
		trainset = data.build_full_trainset()

		start = time.time()
		self.algo.fit(trainset)
		end = time.time()
		# print("Train time: {}".format(end - start))

	def surprise_predict(self):
		# print("Predicting...")
		start = time.time()
		predictions = []
		for item_id in self.testset:
			prediction = self.algo.predict(self.user_id, item_id)
			predictions.append(prediction)
		end = time.time()
		# print("Prediction Time: {}".format(end-start))
		return predictions
