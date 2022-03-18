import numpy as np
import scipy as sp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score



def flatten_dim12(data):
    num_epochs = data.shape[0]
    num_channels = data.shape[1]
    num_samples = data.shape[2]

    data_flattened = data.reshape(num_epochs, num_channels * num_samples)
    return data_flattened


def revolving_window(data, labels, min_data_length=500, max_duplication=100000):
    
    # Create more data by making a revolving window
    center = data.shape[2] // 2
    offset_amount = 1
    
    # There is definitely a more efficient way to do this
    # windowed_data_length = int((data.shape[2] - min_data_length) / 2 / offset_amount * data.shape[0])
    # window_data = np.zeros((windowed_data_length, data.shape[1], data.shape[2]))
    # window_labels = np.zeros(windowed_data_length)
    window_data = []
    window_labels = []
    
    # Data of label 1
    data_ones = data[labels == 1]
    data_zeros = data[labels == 0]
    
    # Window the ones data to squeeze out as much as we can
    for i in range(len(data_ones)):
        offset = data.shape[2] // 2
        data_length = 2 * offset
        j = 0
        if j >= max_duplication:
            break
        while (data_length > min_data_length and j < max_duplication):
            window_labels.append(1)
            window = np.full((data.shape[1], data.shape[2]), 0)
            window[:, center-offset:center+offset] = data[i][:, center-offset:center+offset]
            offset = offset - offset_amount
            data_length = 2 * offset
            window_data.append(window)
            j += 1
    
    # Fill out the rest of the windowed data with the zero event data until we reach an equal split of classes
    num_needed = len(window_data)
    for i in range(len(data_zeros)):
        offset = data.shape[2] // 2
        data_length = 2 * offset
        label = labels[i]
        j = 0
        if j >= max_duplication:
            break
        while (num_needed > 0 and data_length > min_data_length and j < max_duplication):
            window_labels.append(0)
            window = np.full((data.shape[1], data.shape[2]), 0)
            window[:, center-offset:center+offset] = data[i][:, center-offset:center+offset]
            offset = offset - offset_amount
            data_length = 2 * offset
            window_data.append(window)
            num_needed = num_needed - 1
            j += 1
    
    window_data = np.array(window_data)
    window_labels = np.array(window_labels)
    
    return window_data, window_labels


def lstm(train_data, train_labels, test_data):
    
    # one hot encode the training labels
    y_train = keras.utils.to_categorical(train_labels, num_classes=2)
    
    ## Build model
    model = keras.Sequential()
    embedding_length = train_data.shape[-1]
    embedding_channels = train_data.shape[-2]
    model.add(keras.layers.LSTM(units=100, input_shape=(embedding_channels, embedding_length)))
    model.add(Dropout(0.2))
    
    # Predict a binary outcome (0 or 1)
    model.add(Dense(2, activation='sigmoid'))

    ## Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    ## Fit model
    history = model.fit(train_data, y_train, epochs=10, batch_size=64, verbose=1)
    
    ## Evaluate model
    train_predictions = model.predict(train_data)
    test_predictions = model.predict(test_data)
    
    return train_predictions, test_predictions, history


def baseline(train_data, train_labels, test_data):

    # flatten EEG data
    train_data = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
    test_data = test_data.reshape((test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
    y_train = keras.utils.to_categorical(train_labels, num_classes=2)

    model = keras.Sequential()
    model.add(Dense(60, activation='relu', input_dim=train_data.shape[1]))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build()

    print(model.summary())
    history = model.fit(train_data, y_train, epochs=10, batch_size=64, verbose=1)

    ## Evaluate model
    train_predictions = model.predict(train_data)
    test_predictions = model.predict(test_data)
    
    return train_predictions, test_predictions, history


def random_forest(train_data, train_labels, test_data, n_estimators=1000, random_state=42):

    train_data_ = flatten_dim12(train_data)
    test_data_ = flatten_dim12(test_data)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(train_data_, train_labels)
    train_predictions = clf.predict(train_data_)
    test_predictions = clf.predict(test_data_)
    
    return train_predictions, test_predictions
    
    
def svm(train_data, train_labels, test_data):

    train_data_ = flatten_dim12(train_data)
    test_data_ = flatten_dim12(test_data)

    clf = SVC(kernel='linear')
    clf = clf.fit(train_data_, train_labels)
    train_predictions = clf.predict(train_data_)
    test_predictions = clf.predict(test_data_)
    
    return train_predictions, test_predictions
    
    
def lda(train_data, train_labels, test_data):

    train_data_ = flatten_dim12(train_data)
    test_data_ = flatten_dim12(test_data)

    clf = LinearDiscriminantAnalysis()
    clf.fit(train_data_,train_labels)
    train_predictions = clf.predict(train_data_)
    test_predictions = clf.predict(test_data_)
    
    return train_predictions, test_predictions


def CSP(c1_data, c2_data, n_top):
    
    num_c1_trials = c1_data.shape[0]
    num_c2_trials = c2_data.shape[0]
    num_channels = c1_data.shape[1]
    
    ## Calculate normalized spatial covariance of each trial
    c1_trial_covs = np.zeros((num_c1_trials, num_channels, num_channels))
    c2_trial_covs = np.zeros((num_c2_trials, num_channels, num_channels))
    
    for i in range(num_c1_trials):
        c1_trial = c1_data[i]
        c1_trial_prod = c1_trial @ c1_trial.T
        c1_trial_cov = c1_trial_prod / (np.trace(c1_trial_prod))
        c1_trial_covs[i] = c1_trial_cov
    
    for i in range(num_c2_trials):
        c2_trial = c2_data[i]
        c2_trial_prod = c2_trial @ c2_trial.T
        c2_trial_cov = c2_trial_prod / (np.trace(c2_trial_prod))
        c2_trial_covs[i] = c2_trial_cov
    
    
    ## Calculate averaged normalized spatial covariance
    c1_trial_covs_avg = np.mean(c1_trial_covs, axis=0)
    c2_trial_covs_avg = np.mean(c2_trial_covs, axis=0)
    
    ## Calculate composite spatial covariance
    R12 = c1_trial_covs_avg + c2_trial_covs_avg
    
    ## Eigen-decompose composite spatial covariance
    R12_eigval, R12_eigvec = np.linalg.eig(R12)
    
    ## Create diagonal matrix of eigenvalues        
    R12_eigval_diag = np.diag(R12_eigval)
    
    ## Calculate Whitening transformation matrix
    P12 = np.linalg.inv(np.sqrt(R12_eigval_diag)) @ R12_eigvec.T
    
    ## Whitening Transform average covariance
    S12_1 = P12 @ c1_trial_covs_avg @ P12.T
    S12_2 = P12 @ c2_trial_covs_avg @ P12.T
    
    ## Eigen-decompose whitening transformed average covariance
    S12_1_eigval, S12_1_eigvec = np.linalg.eig(S12_1)
    S12_2_eigval, S12_2_eigvec = np.linalg.eig(S12_2)
    
    #print(S12_1_eigval + S12_2_eigval)
    
    ## Take the top and bottom eigenvectors to contruct projection matrix W
    sort_indices = np.argsort(S12_1_eigval)
    top_n_indices = list(sort_indices[-n_top:])
    bot_n_indices = list(sort_indices[:n_top])
    S12_1_eigvec_extracted = S12_1_eigvec[:, top_n_indices + bot_n_indices]
    W12 = S12_1_eigvec_extracted.T @ P12
    
    return W12


def CSP_extract_features(all_data, W, n_top):

    extracted_features = np.zeros((all_data.shape[0], 2 * n_top))

    for i, epoch in enumerate(all_data):

        Z = W @ epoch

        #print(np.var(Z, axis=-1).shape)

        var_sum = np.sum(np.var(Z, axis=-1))

        for k in range(n_top):

            Z_k = Z[k]
            f_k = np.log10(np.var(Z_k) / var_sum)

            extracted_features[i, k] = f_k

    return extracted_features


def csp_lda(train_data, train_labels, test_data, n_top=1):

    ## Separate data by labels
    unique_labels = np.unique(train_labels)
    num_unique_labels = len(unique_labels)

    assert num_unique_labels == 2
        
    data_c1 = train_data[train_labels == unique_labels[0]]
    data_c2 = train_data[train_labels == unique_labels[1]]
    labels_c1 = train_labels[train_labels == unique_labels[0]]
    labels_c2 = train_labels[train_labels == unique_labels[1]]
            
    ## Apply CSP to transform train data
    CSP_transform = CSP(data_c1, data_c2, n_top)
    train_data_ = CSP_extract_features(train_data, CSP_transform, n_top)
    
    ## Train LDA classifier
    clf = LinearDiscriminantAnalysis(solver='lsqr',  shrinkage='auto')
    clf.fit(train_data_, train_labels)
           
    ## Apply CSP to transform test data
    test_data_ = CSP_extract_features(test_data, CSP_transform, n_top)

    ## Make predictions
    train_predictions = clf.predict(train_data_)
    test_predictions = clf.predict(test_data_)
    
    return train_predictions, test_predictions
