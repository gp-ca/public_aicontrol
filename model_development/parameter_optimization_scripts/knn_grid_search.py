#############################################################################
# import required libraries
#############################################################################
from utilities import get_grid_search_data, preprocess_grid_search_data, \
    perform_grid_search, print_loading_bar
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import PredefinedSplit
import itertools

print("Parameter optimization of KNN")
###########################################################################
# KNN parameter optimization using grid search
###########################################################################

# Define classifier
classifier = KNeighborsClassifier()

# Define parameter grid for KNN
param_grid = [
    {
        'classifier__n_neighbors': [3, 5, 10, 20],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__n_jobs': [-1]
    }
]


transformer_options = ['none', 'log', 'arcsinh']
scaler_options = ['none', 'sc_train']
umap_options = [False, True]

data, labels, umap_fit, umap_scaler = get_grid_search_data()

loop = 0
res = []
for transformer, scaler, use_umap in \
     itertools.product(transformer_options, scaler_options, umap_options):
    loop += 1
    # Get data for grid search
    x_train, y_train, x_valid, y_valid, train_indices, valid_indices = \
        preprocess_grid_search_data(fcs_data=data.copy(),
                                    labelset=labels.copy(),
                                    n_samples=50,
                                    train_size=25,
                                    valid_size=25,
                                    event_size=15,
                                    transformer=transformer,
                                    scaler=scaler,
                                    use_umap=use_umap,
                                    umap_fit=umap_fit,
                                    umap_scaler=umap_scaler,
                                    random_state=28)

    # Aggregate data
    X = pd.concat([pd.concat(x_train), pd.concat(x_valid)])
    y = pd.concat([pd.concat(y_train), pd.concat(y_valid)])

    # Use PredefinedSplit to make use of curated train_test_split
    test_fold = np.zeros(len(X))
    test_fold[:len(pd.concat(x_train))] = -1
    ps = PredefinedSplit(test_fold=test_fold)
    print_loading_bar(iteration=loop,
                      total=(len(transformer_options) * len(scaler_options) *
                             len(umap_options)),
                      length=50)
    score, params, model = perform_grid_search(X=X, y=y, classifier=classifier,
                                               param_grid=param_grid, ps=ps)
    res.append(pd.DataFrame(
        [[score, params, transformer, scaler, str(use_umap)]],
        columns=['score', 'params', 'transformer', 'scaler', 'umap']))

knn_grid_search_results = pd.concat(res)
