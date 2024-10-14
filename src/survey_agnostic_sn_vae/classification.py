import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
#import torch
#import torch.nn as nn
#import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from survey_agnostic_sn_vae.hxe_utils import *
import h5py
from survey_agnostic_sn_vae.data_imports.import_ztf import canonicalize
from survey_agnostic_sn_vae.autoencoder.raenn_equinox import evaluate, VAE
import equinox as eqx

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
    model_fn,
    data_fn,
    data_type='4way'
):
    """Retrieve data from outfile.
    Includes LCs from surveys listed by include_surveys.
    If remove_duplicates is True, prioritizes joint > YSE > ZTF
    light curves, and removes other repeats.
    """
    with h5py.File(data_fn, "r") as f:
        classes = f['classes'][:]
        ids = f['ids'][:]
        types = np.array([canonicalize(c.decode('utf-8')) for c in classes])
        print(np.unique(classes, return_counts=True))
        encoder_inputs = f['encoder_input'][:]
        num_samples = len(encoder_inputs)
        #val_inputs = encoder_inputs[num_samples // 10 * 9:]
        #types = classes[num_samples // 10 * 9:]


    # TODO: de-hardcode this
    config = {
        "hidden_dim": 16,
        "out_dim": 4
    }
    model = eqx.filter_eval_shape(VAE, **config)
    model = eqx.tree_deserialise_leaves(model_fn, model)

    out_dir = evaluate(model, encoder_inputs)
    z_means = out_dir['mu']
    z_logvars = out_dir['logvar']    

    if data_type == '4way':
        type_list = ['SN Ia', 'SN Ia-norm', 'SN II', 'SN IIn', 'SLSN-II', 'SN Ibc', 'SN Ib', 'SN Ic', 'SN Ic-BL']
        gind = np.isin(types, type_list)

        # Assuming data['z_means'][gind], data['z_logvars'][gind], and types[gind] are defined
        features = z_means[gind]
        features_err = z_logvars[gind]
        types = types[gind]
        ids = ids[gind]

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
        
    elif data_type == '3way':
        type_list = ['SN Ia', 'SN Ia-norm', 'SN II', 'SN Ibc', 'SN Ib', 'SN Ic', 'SN Ic-BL']
        gind = np.isin(types, type_list)

        # Assuming data['z_means'][gind], data['z_logvars'][gind], and types[gind] are defined
        features = z_means[gind]
        features_err = z_logvars[gind]
        types = types[gind]
        ids = ids[gind]

        # Clean up some labels...
        gind_ia = np.where(types == 'SN Ia-norm')
        types[gind_ia] = 'SN Ia'
        gind_ib = np.where(types == 'SN Ib')
        types[gind_ib] = 'SN Ibc'
        gind_ic = np.where(types == 'SN Ic')
        types[gind_ic] = 'SN Ibc'
        gind_icbl = np.where(types == 'SN Ic-BL')
        types[gind_icbl] = 'SN Ibc'
        #gind_iin = np.where(types == 'SN IIn')
        #types[gind_iin] = 'SN II'
        
    elif data_type == '5way':
        type_list = ['SLSN-I', 'SN Ia', 'SN Ia-norm', 'SN II', 'SN IIn', 'SLSN-II', 'SN Ibc', 'SN Ib', 'SN Ic', 'SN Ic-BL']
        gind = np.isin(types, type_list)

        # Assuming data['z_means'][gind], data['z_logvars'][gind], and types[gind] are defined
        features = z_means[gind]
        features_err = z_logvars[gind]
        types = types[gind]
        ids = ids[gind]

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
    return features, features_err, types, ids


def vanilla_rf(features, features_err, types, ids):
    # Combining features and features_err into one set of features
    X = np.concatenate((features, features_err), axis=1)
    y = types

    # Splitting the data into training and testing sets
    y_pred_comb = np.array([])
    y_true_comb = np.array([])
    ids_comb = np.array([])
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ids_train, ids_test = ids[train_index], ids[test_index]

        # Creating and training the random forest classifier
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)

        # Predicting the test set results
        y_pred = clf.predict(X_test)
        y_pred_comb = np.append(y_pred_comb, y_pred)
        y_true_comb = np.append(y_true_comb, y_test)
        ids_comb = np.append(ids_comb, ids[test_index])

    return y_true_comb, y_pred_comb, ids_comb, clf.classes_

def disp_cm(y_true, y_pred, class_mapping):
    # Creating the confusion matrix

    acc, f1 = calc_acc_f1_score(y_true, y_pred, class_mapping)

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm_purity = confusion_matrix(y_true, y_pred, normalize='pred')

    # Displaying the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_mapping)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Completeness, A={round(acc,3)}, F1={round(f1,3)}")
    plt.show()
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_purity, display_labels=class_mapping)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Purity, A={round(acc,3)}, F1={round(f1,3)}")
    plt.show()

def vanilla_gbm(features, features_err, types):
    # Combining features and features_err into one set of features
    X = np.concatenate((features, features_err), axis=1)
    y = types

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Creating and training the random forest classifier
    clf = LGBMClassifier(n_estimators=1000, class_weight='balanced', random_state=42)
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
"""
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

    print(X_train.shape, X_test.shape)

    model = SimpleMLP(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    epochs = 10 # you'll probably need more to get good results but it's kinda slow
    batch_size = 1
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])

        # https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way
        for i in range(0,X_train.size()[0], batch_size):
            model.train()
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            outputs = model(batch_x)
            loss = criterion(outputs,batch_y)

            loss.backward()
            optimizer.step()
        print(loss)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predicted = []
        for i in range(X_test.shape[0]):
            outputs = model(X_test[i,:])
            predicted.append(torch.argmax(outputs))

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
"""
    

def hxe(features, features_err, types):
    paths, pathlengths, mask_list, y_dict = calc_path_and_mask(G, vertices, 'Object')
    class_weight_dict = calc_class_weights(types, vertices, paths)

    labels_new = [y_dict[x] for x in types]
    weights = [class_weight_dict[x] for x in types]
    X_train, X_test, y_train, y_test, labels_train, labels_test, weights_train, weights_test = train_test_split(features, labels_new, types, weights,test_size=0.33)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    weights_train = torch.tensor(weights_train, dtype=torch.float32)
    weights_test = torch.tensor(weights_test, dtype=torch.float32)

    # train model
    model = Feedforward(np.shape(X_test[0])[0],15, len(vertices))
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
    model.train()
    epochs = 20 # you'll probably need more to get good results but it's kinda slow
    batch_size = 1 # I get a segfault if this >1
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])

        # https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way
        for i in range(0,X_train.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y, batch_weights = X_train[indices], y_train[indices], weights_train[indices]

            outputs = model(batch_x)
            loss = custom_hier_loss(outputs,batch_y, batch_weights, mask_list, pathlengths)

            loss.backward()
            optimizer.step()
        print(loss)

    # get predictions
    model.eval()
    with torch.no_grad():
        predicted = []
        for i in range(X_test.shape[0]):
            outputs = model(X_test[i,:])
            predicted.append(outputs.numpy())

    # get classifications just in the types we need
    leaves = np.unique(types)
    probs_list = np.zeros((len(predicted), len(leaves)))
    for i,leaf in enumerate(leaves):
        probs_list[:,i] = get_prob((np.array(predicted)),leaf, paths, vertices, np.array(mask_list)).detach().numpy()
    my_predicted_types = leaves[np.argmax(probs_list,axis=1)]

    cm = confusion_matrix(labels_test, my_predicted_types, normalize='true')
    cm_purity = confusion_matrix(labels_test, my_predicted_types,  normalize='pred')

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=leaves)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Completeness Matrix")
    plt.show()
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_purity, display_labels=leaves)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Purity Matrix")
    plt.show()

"""
if __name__ == '__main__':
    features, features_err, types = get_data("/Users/kdesoto/python_repos/survey-agnostic-sn-vae/data/superraenn/yse/outputs/out.npz", data_type = '4way')
    vanilla_rf(features, features_err, types)
    vanilla_mlp(features, features_err, types)
    hxe(features, features_err, types)
"""