import numpy as np
import dataSetFeatureExtractor

import sklearn
import sklearn.datasets
import sklearn.preprocessing
import sklearn.tree
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.neighbors
import sklearn.ensemble
import sklearn.model_selection

from DecsionTree import decisonTree


def build_table(x, y, boxes, black_box_model, meta_features):

    """
    :param x: 2-d numpy matrix
    :param y: 1-d numpy array
    :param boxes: list of list of index of row in each box
    :param black_box_model: model object that has .predict()
    :param meta_features: the list of meta-features we want to use
    :return: meta_x, which is meta-feature for each box
             meta_y, which is the min error for each box
    """

    meta_x = np.zeros((len(boxes), len(meta_features)))
    meta_y = np.zeros(len(boxes))

    for i in range(len(boxes)):
        box_x = x[boxes[i]]
        box_y = y[boxes[i]]

        extractor = dataSetFeatureExtractor.DataSetFeatureExtractor(box_x, box_y)
        meta_x[i] = extractor.extract_features(meta_features)
        meta_y[i] = get_min_error(box_x, black_box_model)
    return meta_x, meta_y


def get_min_error(x, black_box_model):
    accuracies = list()

    # We use y_hat as the label
    y_hat = black_box_model.predict(x)

    if len(np.unique(y_hat)) == 1:
        return 0.

    n_split = 2
    kf = sklearn.model_selection.StratifiedKFold(n_splits=n_split)

    simple_models = [sklearn.tree.DecisionTreeClassifier(max_depth=3),
                     sklearn.linear_model.LogisticRegression(penalty='l1')]

    for model in simple_models:
        accuracy = 0.

        for train, test in kf.split(x, y_hat):
            # If in this box, y only has 1 label, we deal with this separately
            if len(np.unique(y_hat[train])) != 1:
                model.fit(x[train], y_hat[train])
                prediction = model.predict(x[test])
                # print sklearn.metrics.accuracy_score(prediction, y_hat[test])

                accuracy += sklearn.metrics.accuracy_score(prediction, y_hat[test]) / n_split

        accuracies.append(accuracy)

    return 1 - max(accuracies)


def build_min_error_predictor(x, y):

    """
    Min error predictor based on the meta-feature table
    :param x: meta-features
    :param y: simple model prediction error
    :return: model
    """

    min_error_predictor = sklearn.ensemble.RandomForestRegressor(n_estimators=20)
    min_error_predictor.fit(x, y)
    print "min error predictor training r^2:", min_error_predictor.score(x, y)

    return min_error_predictor


# test case
