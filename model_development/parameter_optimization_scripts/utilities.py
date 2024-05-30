import sys
import copy
import numpy as np
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from umap.parametric_umap import load_ParametricUMAP
import contextlib
import io
import tensorflow as tf


# Pseudo-log transform
def log_transform(x):
    return np.log(np.abs(x) + 1) * np.sign(x)


# Arcsinh transform
def arcsinh_transform(x):
    return np.arcsinh(x)


def load_parametricUMAP_silently(file_path):
    # Funtion to load parametric umap model without warnings beeing printed to
    # the console
    original_log_level = tf.get_logger().level
    tf.get_logger().setLevel('ERROR')
    with contextlib.redirect_stdout(io.StringIO()) as f, \
         contextlib.redirect_stderr(io.StringIO()):
        umap_model = load_ParametricUMAP(file_path)
    tf.get_logger().setLevel(original_log_level)
    return umap_model


def get_grid_search_data():
    raw_data_preprocessed = pd.read_pickle(
        pathlib.Path(r'..\\data\\processed_data\\raw_data_preprocessed.pkl'))
    # Get list of unique indices
    indices = raw_data_preprocessed["measurement_data_compensated"].\
        index.get_level_values(0).unique()
    # Use groupby to split DataFrame
    grouped = raw_data_preprocessed["measurement_data_compensated"].\
        groupby(level=0)
    # get fcs data as a list of dataframes of each sample
    fcs_data = [grouped.get_group(index).reset_index(drop=True) for index in
                indices]
    # same for labels
    indices = raw_data_preprocessed["reported_labels"].index.\
        get_level_values(0).unique()
    grouped = raw_data_preprocessed["reported_labels"].groupby(level=0)
    labelset = [grouped.get_group(index).reset_index(drop=True) for index in
                indices]
    umap_fit = load_parametricUMAP_silently(
        pathlib.Path(r'saved_models\\seq_nn_parametric_umap')
    )
    umap_scaler = joblib.load(pathlib.Path(r'scalers\\umap_scaler.save'))
    return fcs_data, labelset, umap_fit, umap_scaler


def select_random_indices(df, num_samples, cell_subsets_of_interest,
                          random_state=0):
    """
    This function selects random indices from a pandas DataFrame where a
    specific condition is met
    """
    np.random.seed(random_state)
    conditions = [
        [0, 0, 0, 0, 0, 0],  # only and exactly none
        [1, 0, 0, 0, 0, 0],  # only and exactly lympho
        [1, 1, 0, 0, 0, 0],  # only and exactly BP
        [1, 0, 1, 0, 0, 0],  # only and exactly NKP
        [1, 0, 0, 1, 0, 0],  # only and exactly TP
        [1, 0, 0, 1, 1, 0],  # only and exactly T4P
        [1, 0, 0, 1, 0, 1]  # only and exactly T8P
    ]

    selected_rows = pd.DataFrame()
    for condition in conditions:
        condition_met = np.array(
            df[cell_subsets_of_interest] == condition
        ).all(axis=1)
        true_indices = np.where(condition_met)[0]
        selected_indices = np.random.choice(
            true_indices, size=num_samples, replace=False
        )
        selected_rows = pd.concat([selected_rows, df.iloc[selected_indices]])

    return selected_rows


def preprocess_grid_search_data(fcs_data=None, labelset=None, n_samples=50,
                                train_size=25, valid_size=25, event_size=10,
                                transformer='none',
                                scaler='none',
                                fluorescent_indices=['FITC-A', 'PE-A',
                                                     'PerCP-A', 'PE-Cy7-A',
                                                     'APC-A', 'APC-H7-A',
                                                     'Pacific Blue-A',
                                                     'AmCyan-A'],
                                cell_subsets_of_interest=['Lympho', 'BP',
                                                          'NKP', 'TP', 'T4P',
                                                          'T8P'],
                                use_umap=False,
                                umap_fit=None,
                                umap_scaler=None,
                                random_state=1):
    ##########################################################################
    # Somewhat overspecified function loading grid search data
    ##########################################################################
    fcs_data = copy.deepcopy(fcs_data[:n_samples])
    labelset = copy.deepcopy(labelset[:n_samples])

    if isinstance(fcs_data[0], pd.DataFrame):
        colnames = fcs_data[0].columns.tolist()
    else:
        raise ValueError("\"raw_data_preprocessed.pkl\" is required")

    # check if colnames match requirement
    if colnames != ['FSC-A', 'SSC-A', 'FITC-A', 'PE-A',
                    'PerCP-A', 'PE-Cy7-A', 'APC-A', 'APC-H7-A',
                    'Pacific Blue-A', 'AmCyan-A']:
        raise ValueError("Column names do not match. The data used is \
                         currently not supported")

    # Apply umap transform if keyword is True
    if use_umap is True:
        colnames += ['umap1', 'umap2']
        for i in range(0, n_samples):
            X_embedded = umap_fit.transform(umap_scaler.transform(fcs_data[i]))
            fcs_data[i] = pd.DataFrame(np.hstack((fcs_data[i], X_embedded)),
                                       columns=colnames)

    # Logic for transformation of fluorescent channels
    if transformer == 'log':
        for i in range(0, n_samples):
            fcs_data[i][fluorescent_indices] = log_transform(
                fcs_data[i][fluorescent_indices]
                )
            fcs_data[i] = pd.DataFrame(fcs_data[i], columns=colnames)
    elif transformer == 'arcsinh':
        for i in range(0, n_samples):
            fcs_data[i][fluorescent_indices] = arcsinh_transform(
                fcs_data[i][fluorescent_indices]
                )
            fcs_data[i] = pd.DataFrame(fcs_data[i], columns=colnames)

    ##########################################################################
    # train test split
    ##########################################################################
    # Split data and keep track of indices
    indices = range(len(fcs_data))
    train_indices, valid_indices = train_test_split(
        indices, train_size=train_size, test_size=valid_size,
        random_state=random_state
    )
    x_train = [fcs_data[i] for i in train_indices]
    y_train = [labelset[i][cell_subsets_of_interest] for i in train_indices]
    x_valid = [fcs_data[i] for i in valid_indices]
    y_valid = [labelset[i][cell_subsets_of_interest] for i in valid_indices]

    # select n cells for each cell subset of interest
    y_train_sample = [
        select_random_indices(df[cell_subsets_of_interest], event_size,
                              cell_subsets_of_interest, random_state)
        for df in y_train
        ]

    x_train_sample = [
        x.iloc[y.index] for x, y in zip(x_train, y_train_sample)
    ]

    # select n cells for each cell subset of interest
    y_valid_sample = [
        select_random_indices(df[cell_subsets_of_interest], event_size,
                              cell_subsets_of_interest, random_state)
        for df in y_valid
        ]

    x_valid_sample = [
        x.iloc[y.index] for x, y in zip(x_valid, y_valid_sample)
    ]

    if scaler == 'sc_train':
        scaler = StandardScaler().fit(pd.concat(x_train_sample))
        joblib.dump(scaler, "scalers\\grid_train_scaler.save")
        for i in range(0, len(x_train_sample)):
            x_train_sample[i] = pd.DataFrame(
                scaler.transform(x_train_sample[i]), columns=colnames)
            x_valid_sample[i] = pd.DataFrame(
                scaler.transform(x_valid_sample[i]), columns=colnames)

    return x_train_sample, y_train_sample, x_valid_sample, y_valid_sample, \
        train_indices, valid_indices


# Function perform grid search
def perform_grid_search(X, y, classifier, param_grid, ps):
    pipeline = Pipeline(
        [('classifier', classifier)]
        )
    grid_search = GridSearchCV(pipeline, param_grid, cv=ps, scoring="f1_macro",
                               n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_score_, grid_search.best_params_, \
        grid_search.best_estimator_


# for unknown reasons, perform_grid_search takes ages for the OvR SVC
def perform_grid_search_SVM(x_train, y_train, x_valid, y_valid, param_grid):
    # Initialize variables to store best results
    best_score = 0
    best_params = None
    best_model = None

    # Iterate over all combinations of parameters
    for params in param_grid:
        for C in params['classifier__estimator__C']:
            for kernel in params['classifier__estimator__kernel']:
                for gamma in params['classifier__estimator__gamma']:

                    classifier = OneVsRestClassifier(
                        SVC(C=C, kernel=kernel, gamma=gamma)
                    )

                    classifier.fit(x_train, y_train)

                    y_pred = pd.DataFrame(
                        classifier.predict(x_valid),
                        columns=y_valid.columns
                    )
                    score = f1_score(
                        y_true=y_valid,
                        y_pred=y_pred,
                        average='macro'
                    )

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'classifier__estimator__C': C,
                            'classifier__estimator__kernel': kernel,
                            'classifier__estimator__gamma': gamma
                        }
                        best_model = classifier

    return best_score, best_params, best_model


def print_loading_bar(iteration, total, length=50):
    """
    Prints a loading bar for the given iteration and total.
    :param iteration: Current iteration
    :param total: Total number of iterations
    :param length: Length of the loading bar (default is 50)
    """
    progress = (iteration / total)
    arrow = '=' * int(round(progress * length) - 1)
    spaces = ' ' * (length - len(arrow))
    sys.stdout.write(f"\rStarting grid search ... [{arrow}{spaces}] \
                     {int(progress * 100)}%")
    sys.stdout.flush()
