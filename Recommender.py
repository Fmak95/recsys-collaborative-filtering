from jikanpy import Jikan
import pandas as pd
import numpy as np
import surprise


# Recommender handles the machine learning to recommend people new anime to watch.
# The data used for the machine learning was scraped from https://myanimelist.net/

class Recommender():
	def __init__(self, data_filepath, username):
		self.username = username
		self.data_filepath = data_filepath
		self.data = self.prep_data()

	def prep_data(self):
		