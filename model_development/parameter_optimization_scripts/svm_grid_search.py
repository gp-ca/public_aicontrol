#############################################################################
# import required libraries
#############################################################################
from utilities import get_grid_search_data, preprocess_grid_search_data, \
    perform_grid_search_SVM, print_loading_bar
import pandas as pd
import itertools

print("Parameter optimization of OneVsRestClassifier SVM")
###########################################################################
# SVM parameter optimization using grid search
###########################################################################

# Define classifier
# classifier = OneVsRestClassifier(SVC())

# Define parameter grid for SVM
param_grid = [
    {
        'classifier__estimator__C': [1, 2, 3, 4, 5],
        'classifier__estimator__kernel': ['rbf'],
        'classifier__estimator__gamma': ['scale', 'auto', 0.1]
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
    X_train = pd.concat(x_train)
    X_valid = pd.concat(x_valid)
    y_train = pd.concat(y_train)
    y_valid = pd.concat(y_valid)

    print_loading_bar(iteration=loop,
                      total=(len(transformer_options) * len(scaler_options) *
                             len(umap_options)),
                      length=50)

    score, params, model = perform_grid_search_SVM(
        x_train=X_train,
        y_train=y_train,
        x_valid=X_valid,
        y_valid=y_valid,
        param_grid=param_grid

    )
    res.append(pd.DataFrame(
        [[score, params, transformer, scaler, str(use_umap)]],
        columns=['score', 'params', 'transformer', 'scaler', 'umap']))

svm_grid_search_results = pd.concat(res)
