import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score
import tqdm

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.htm
def logreg(X_train, y_train, X_test, y_test, max_iter=1000, grid_search=False, predefined_val_indices=None, standardize=False, normalize=True):

    if standardize:
        scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    if normalize:
        X_train = sklearn.preprocessing.normalize(X_train, norm='l2')
        X_test = sklearn.preprocessing.normalize(X_test, norm='l2')

    clf = LogisticRegression(
        penalty='l2',
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        class_weight=None,
        solver='lbfgs',
        max_iter=max_iter,
        multi_class='multinomial',
        warm_start=True, # GVH: GridSearch does NOT use this.
        n_jobs=-1
    )

    if grid_search:

        C_values = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]
        parameters = {'C' : C_values}

        if predefined_val_indices is not None:
            cv = PredefinedSplit(test_fold=predefined_val_indices)
        else:
            cv = 3
        clf = GridSearchCV(clf, parameters, n_jobs=-1, cv=cv, refit=True)

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results = {
        'acc' : accuracy_score(y_test, y_pred),
    }

    if grid_search:
        results['best_param'] = clf.best_params_['C']

    return results


# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
def linearsvc(X_train, y_train, X_test, y_test, max_iter=1000, grid_search=False, predefined_val_indices=None, standardize=False, normalize=True, dual=False):
    """
    """

    if standardize:
        scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    if normalize:
        X_train = sklearn.preprocessing.normalize(X_train, norm='l2')
        X_test = sklearn.preprocessing.normalize(X_test, norm='l2')

    clf = LinearSVC(
        random_state=0,
        tol=1e-5,
        C=1.,
        dual=dual,
        class_weight=None,
        max_iter=max_iter
    )

    if grid_search:

        C_values = [0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]
        parameters = {'C' : C_values}

        if predefined_val_indices is not None:
            cv = PredefinedSplit(test_fold=predefined_val_indices)
        else:
            cv = 3
        clf = GridSearchCV(clf, parameters, n_jobs=-1, cv=cv, refit=True)

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results = {
        'acc' : accuracy_score(y_test, y_pred)
    }

    if grid_search:
        results['best_param'] =  clf.best_params_['C']

    return results


# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
def sgd(X_train, y_train, X_test, y_test, max_iter=1000, loss_type='hinge', grid_search=False, predefined_val_indices=None, standardize=False, normalize=True):

    if standardize:
        scaler = sklearn.preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    if normalize:
        X_train = sklearn.preprocessing.normalize(X_train, norm='l2')
        X_test = sklearn.preprocessing.normalize(X_test, norm='l2')


    clf = SGDClassifier(
        loss=loss_type,
        penalty='l2',
        alpha=0.0001,
        fit_intercept=True,
        max_iter=max_iter,
        tol=1e-5,
        shuffle=True,
        random_state=0,
        n_jobs=-1,
        learning_rate='optimal',
        class_weight=None,
        warm_start=True
    )

    if grid_search:

        alpha_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1., 10.]
        parameters = {'alpha' : alpha_values}

        if predefined_val_indices is not None:
            cv = PredefinedSplit(test_fold=predefined_val_indices)
        else:
            cv = 3
        clf = GridSearchCV(clf, parameters, n_jobs=-1, cv=cv, refit=True)

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results = {
        'acc' : accuracy_score(y_test, y_pred)
    }

    if grid_search:
        results['best_param'] = clf.best_params_['alpha']

    return results