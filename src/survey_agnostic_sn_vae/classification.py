import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

def calc_acc_f1_score(y_true, y_pred, labels):
    """Calculate accuracy and F1 scores from
    y_true and y_pred.
    """
    f1 = f1_score(
        y_true, y_pred, labels=labels, average='macro'
    )
    acc = np.sum((y_true == y_pred).astype(int)) / len(y_true)
    return acc, f1


def get_data(
    outfile,
    include_surveys=['YSE', 'ZTF', 'joint'],
    remove_duplicates=True,
    data_type='4way'
):
    """Retrieve data from outfile.
    Includes LCs from surveys listed by include_surveys.
    If remove_duplicates is True, prioritizes joint > YSE > ZTF
    light curves, and removes other repeats.
    """
    data = np.load(outfile)
    include_surveys = np.atleast_1d(include_surveys) # in case a str is provided
    
    surveys = data['surveys']
    survey_submasks = {}
    
    for survey in include_surveys:
        survey_submasks[survey] = surveys == survey
        
    ids = data['ids']
    # now remove redundancies
    if 'joint' in survey_submasks:
        joint_ids = ids[survey_submasks['joint']]
        for k in ['YSE', 'ZTF']:
            if k not in survey_submasks:
                continue
            same_ids = np.isin(ids, joint_ids) & survey_submasks[k]
            survey_submasks[k][same_ids] = False
            
    if ('YSE' in survey_submasks) and ('ZTF' in survey_submasks):
        yse_ids = ids[survey_submask['YSE']]
        same_ids = np.isin(ids, yse_ids) & survey_submasks['ZTF']
        survey_submasks['ZTF'][same_ids] = False
    
    survey_mask = np.zeros(len(surveys)).astype(bool)
    for s in survey_submasks:
        survey_mask = (survey_mask | survey_submasks[s])

    ids = ids[survey_mask]
    types = data['classes'][survey_mask]
    z_means = data['z_means'][survey_mask]
    z_logvars = data['z_logvars'][survey_mask]
    surveys = surveys[survey_mask]
    

    if data_type == '4way':
        type_list = ['SN Ia', 'SN Ia-norm', 'SN II', 'SN IIn', 'SN Ibc', 'SN Ib', 'SN Ic', 'SN Ic-BL']
        gind = np.isin(types, type_list)

        # Assuming data['z_means'][gind], data['z_logvars'][gind], and types[gind] are defined
        features = z_means[gind]
        features_err = z_logvars[gind]
        types = types[gind]

        # Clean up some labels...
        gind_ia = np.where(types == 'SN Ia-norm')
        types[gind_ia] = 'SN Ia'
        gind_ib = np.where(types == 'SN Ib')
        types[gind_ib] = 'SN Ibc'
        gind_ic = np.where(types == 'SN Ic')
        types[gind_ic] = 'SN Ibc'
        gind_icbl = np.where(types == 'SN Ic-BL')
        types[gind_icbl] = 'SN Ibc'
        
    elif data_type == '3way':
        type_list = ['SN Ia', 'SN Ia-norm', 'SN II', 'SN IIn', 'SN Ibc', 'SN Ib', 'SN Ic', 'SN Ic-BL']
        gind = np.isin(types, type_list)

        # Assuming data['z_means'][gind], data['z_logvars'][gind], and types[gind] are defined
        features = z_means[gind]
        features_err = z_logvars[gind]
        types = types[gind]

        # Clean up some labels...
        gind_ia = np.where(types == 'SN Ia-norm')
        types[gind_ia] = 'SN Ia'
        gind_ib = np.where(types == 'SN Ib')
        types[gind_ib] = 'SN Ibc'
        gind_ic = np.where(types == 'SN Ic')
        types[gind_ic] = 'SN Ibc'
        gind_icbl = np.where(types == 'SN Ic-BL')
        types[gind_icbl] = 'SN Ibc'
        gind_iin = np.where(types == 'SN IIn')
        types[gind_iin] = 'SN II'
        
    elif data_type == '5way':
        type_list = ['SLSN-I', 'SN Ia', 'SN Ia-norm', 'SN II', 'SN IIn', 'SLSN-II', 'SN Ibc', 'SN Ib', 'SN Ic', 'SN Ic-BL']
        gind = np.isin(types, type_list)

        # Assuming data['z_means'][gind], data['z_logvars'][gind], and types[gind] are defined
        features = z_means[gind]
        features_err = z_logvars[gind]
        types = types[gind]

        # Clean up some labels...
        gind_ia = np.where(types == 'SN Ia-norm')
        types[gind_ia] = 'SN Ia'
        gind_ib = np.where(types == 'SN Ib')
        types[gind_ib] = 'SN Ibc'
        gind_ic = np.where(types == 'SN Ic')
        types[gind_ic] = 'SN Ibc'
        gind_icbl = np.where(types == 'SN Ic-BL')
        types[gind_icbl] = 'SN Ibc'
        gind_slsnii = np.where(types == 'SLSN-II')
        types[gind_slsnii] = 'SN IIn'
        
    print(np.unique(types, return_counts=True))
    return features, features_err, types


def vanilla_rf(features, features_err, types):
    # Combining features and features_err into one set of features
    X = np.concatenate((features, features_err), axis=1)
    y = types

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Creating and training the random forest classifier
    clf = RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    # Predicting the test set results
    y_pred = clf.predict(X_test)
    
    acc, f1 = calc_acc_f1_score(y_test, y_pred, clf.classes_)

    # Creating the confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    cm_purity = confusion_matrix(y_test, y_pred, normalize='pred')

    # Displaying the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Completeness, A={round(acc,3)}, F1={round(f1,3)}")
    plt.show()
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_purity, display_labels=clf.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Purity, A={round(acc,3)}, F1={round(f1,3)}")
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
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    epochs = 500_000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10000 == 0:
            print(epoch, loss)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, predicted, normalize='true')
    cm_purity = confusion_matrix(y_test, predicted, normalize='pred')

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Completeness Matrix")
    plt.show()
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_purity, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Purity Matrix")
    plt.show()
    
# MAKING HXE WORK NEXT

if __name__ == '__main__':
    features, features_err, types = get_data(data_type = '4way')
    vanilla_rf(features, features_err, types)
    vanilla_mlp(features, features_err, types)