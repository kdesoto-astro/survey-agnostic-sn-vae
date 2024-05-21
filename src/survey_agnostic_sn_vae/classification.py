import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

# Some outstanding issues...
# 1 -- can't identify survey of each LC. Can't fix
# 2 -- adding in HXE. Seems pointless unless 5-way is working.

# keys for out.npz: "ids", "z_means", "z_logvars", "decodings"

def get_data(outfile = 'out_2024-05-15_4_100.npz', sn_file='all_sne.dat', lc_file = 'lcs-2.npz',
			data_type = '4way'):
	data = np.load(outfile)
	lc_data = np.load('lcs-2.npz', allow_pickle=True)
	lcs = lc_data['lcs']

	sn_data_table = np.loadtxt(sn_file, skiprows=41, dtype=str)

	types = np.empty(len(data['ids']), dtype="<U10")

	for i, sn in enumerate(data['ids']):
		gind = np.where(sn_data_table[:,0] == sn)[0][0]
		types[i] = sn_data_table[gind, 3]
		print(lcs[i].filters)


	if data_type == '4way':
		gind = np.where((types == 'SNII') | (types == 'SNIa-norm') | (types == 'SNIIn')\
						| (types == 'SNIb') | (types == 'SNIc') | (types == 'SNIc-BL'))


		# Assuming data['z_means'][gind], data['z_logvars'][gind], and types[gind] are defined
		features = data['z_means'][gind]
		features_err = data['z_logvars'][gind]
		types = types[gind]

		# Clean up some labels...
		gind_ib = np.where(types == 'SNIb')
		types[gind_ib] = 'SNIbc'
		gind_ic = np.where(types == 'SNIc')
		types[gind_ic] = 'SNIbc'
		gind_icbl = np.where(types == 'SNIc-BL')
		types[gind_icbl] = 'SNIbc'

	return features, features_err, types


def vanilla_rf(features, features_err, types):
	# Combining features and features_err into one set of features
	X = np.concatenate((features, features_err), axis=1)
	y = types

	# Splitting the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

	# Creating and training the random forest classifier
	clf = RandomForestClassifier(n_estimators=100, random_state=42)
	clf.fit(X_train, y_train)

	# Predicting the test set results
	y_pred = clf.predict(X_test)

	# Creating the confusion matrix
	cm = confusion_matrix(y_test, y_pred)

	# Displaying the confusion matrix
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
	disp.plot(cmap=plt.cm.Blues)
	plt.title("Confusion Matrix")
	plt.show()



class SimpleMLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(SimpleMLP, self).__init__()
		self.layer1 = nn.Linear(input_dim, hidden_dim)
		self.layer2 = nn.Linear(hidden_dim, output_dim)
	
	def forward(self, x):
		x = torch.relu(self.layer1(x))
		x = self.layer2(x)
		return x


def vanilla_mlp(features, features_err, types):
	# Combine features and features_err into one set of features
	X = np.concatenate((features, features_err), axis=1)
	
	# Encode the string labels into integers
	le = LabelEncoder()
	y = le.fit_transform(types)
	
	# Split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
	# Convert the data to PyTorch tensors
	X_train = torch.tensor(X_train, dtype=torch.float32)
	X_test = torch.tensor(X_test, dtype=torch.float32)
	y_train = torch.tensor(y_train, dtype=torch.long)
	y_test = torch.tensor(y_test, dtype=torch.long)
	
	# Define model, loss function, and optimizer
	input_dim = X_train.shape[1]
	hidden_dim = 5  # You can adjust the hidden layer size
	output_dim = len(le.classes_)
	
	model = SimpleMLP(input_dim, hidden_dim, output_dim)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	
	# Train the model
	epochs = 100
	for epoch in range(epochs):
		model.train()
		optimizer.zero_grad()
		outputs = model(X_train)
		loss = criterion(outputs, y_train)
		loss.backward()
		optimizer.step()
	
	# Evaluate the model
	model.eval()
	with torch.no_grad():
		outputs = model(X_test)
		_, predicted = torch.max(outputs, 1)
	
	# Compute the confusion matrix
	cm = confusion_matrix(y_test, predicted)
	
	# Display the confusion matrix
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
	disp.plot(cmap=plt.cm.Blues)
	plt.title("Confusion Matrix")
	plt.show()
# MAKING HXE WORK NEXT

features, features_err, types = get_data(data_type = '4way')
vanilla_rf(features, features_err, types)
vanilla_mlp(features, features_err, types)