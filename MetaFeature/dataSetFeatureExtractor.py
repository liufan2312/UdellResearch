from collections import defaultdict

import numpy as np
import scipy.stats
import scipy.sparse as sps
import sklearn.metrics
import sklearn.utils

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.utils import check_array
# from sklearn.multiclass import OneVsRestClassifier


class DataSetFeatureExtractor(object):

    featureList = ['number_of_instances',
                   'log_number_of_instances',
                   'number_of_instances_with_missing_values',
                   'percentage_of_instances_with_missing_values',
                   'number_of_features_with_missing_values',
                   'percentage_of_features_with_missing_values',
                   'number_of_missing_values',
                   'percentage_of_missing_values',
                   'data_set_ratio',
                   'log_data_set_ratio',
                   'inverse_data_set_ratio',
                   'log_inverse_data_set_ratio',
                   'class_probability_min',
                   'class_probability_max',
                   'class_probability_mean',
                   'class_probability_std',
                   'kurtosis_min',
                   'kurtosis_max',
                   'kurtosis_mean',
                   'kurtosis_std',
                   'skewness_min',
                   'skewness_max',
                   'skewness_mean',
                   'skewness_std',
                   'class_entropy',
                   'landmark_lda',
                   'landmark_naive_bayes',
                   'landmark_decision_tree',
                   'landmark_decision_node_learner',
                   'landmark_random_node_learner',
                   'landmark_1nn',
                   'pca_fraction_of_components_for_95_percent_variance',
                   'pca_kurtosis_first_pc',
                   'pca_skewness_first_pc']

    def __init__(self, x, y):

        """
        :param x: the data set with categorical variables one-hot encoded,
        x needs to has at least 2 rows.
        :param y: the labels as a 1d row or column vector, assume no missing in y
        """

        if type(x) is not np.ndarray or type(y) is not np.ndarray:
            # TODO: raise correct error
            print "x or y are not numpy arrays"

        if len(y.shape) != 1:
            # TODO: raise correct error
            print "y has to be a row or column vector"

        # TODO: confirm the shape of y
        self.x = x
        self.y = y

        self.numberOfInstances = x.shape[0]
        self.numberOfFeatures = x.shape[1]  # this number is the number after one-hot encoding

        # PCA calculate can't take NaN or inf entries, so we remove those rows
        mask = ~np.any(~np.isfinite(self.x), axis=1)
        self.x_clean = self.x[mask]
        self.y_clean = self.y[mask]

        # keep track of class occurrence
        occurrence_dict = defaultdict(float)
        for value in self.y:
            occurrence_dict[value] += 1

        self.occurrence_dict = occurrence_dict

        if not sps.issparse(self.x):
            self.missing = ~np.isfinite(self.x)

            self.kurts = []
            self.skews = []

            for i in range(self.numberOfFeatures):
                self.kurts.append(scipy.stats.kurtosis(self.x[:, i]))
                self.skews.append(scipy.stats.skew(self.x[:, i]))

            # for i in range(self.numberOfFeatures):
            #     if not categorical[i]:
            #         kurts.append(scipy.stats.kurtosis(X[:, i]))

        else:
            data = [True if not np.isfinite(item) else False for item in self.x.data]
            self.missing = self.x.__class__((data, self.x.indices, self.x.indptr), shape=self.x.shape, dtype=np.bool)

            self.kurts = []
            self.skews = []

            x_new = self.x.tocsc()
            for i in range(x_new.shape[1]):
                start = x_new.indptr[i]
                stop = x_new.indptr[i + 1]
                self.kurts.append(scipy.stats.kurtosis(x_new.data[start:stop]))
                self.skews.append(scipy.stats.skew(x_new.data[start:stop]))

    def get_feature_list(self):
        return self.featureList

    def extract_features(self, feature_used):
        features = []
        for feature in feature_used:
            if feature in self.featureList:
                features.append(getattr(self, feature)())

            else:
                features.append(None)

        return features

    def number_of_instances(self):
        return float(self.numberOfInstances)

    def log_number_of_instances(self):
        return np.log(self.numberOfInstances)

    def number_of_classes(self):
        if len(self.y.shape) == 2:

            return np.mean([len(np.unique(self.y[:, i])) for i in range(self.y.shape[1])])

        else:

            return float(len(np.unique(self.y)))

    def number_of_features(self):
        return float(self.numberOfFeatures)

    def number_of_instances_with_missing_values(self):
        if not sps.issparse(self.x):
            num_missing = self.missing.sum(axis=1)

            return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

        else:
            new_missing = self.missing.tocsr()
            num_missing = [np.sum(new_missing.data[new_missing.indptr[i]:new_missing.indptr[i + 1]]) for i in range(new_missing.shape[0])]

        return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

    def percentage_of_instances_with_missing_values(self):
        return DataSetFeatureExtractor.number_of_instances_with_missing_values(self) / \
               DataSetFeatureExtractor.number_of_instances(self)

    def number_of_features_with_missing_values(self):
        if not sps.issparse(self.x):
            num_missing = self.missing.sum(axis=0)

            return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

        else:
            new_missing = self.missing.tocsc()
            num_missing = [np.sum(new_missing.data[new_missing.indptr[i]:new_missing.indptr[i + 1]]) for i in range(self.missing.shape[1])]

        return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

    def percentage_of_features_with_missing_values(self):
        return DataSetFeatureExtractor.number_of_features_with_missing_values(self) / \
               DataSetFeatureExtractor.number_of_features(self)

    def number_of_missing_values(self):
        return float(self.missing.sum())

    def percentage_of_missing_values(self):
        return float(DataSetFeatureExtractor.number_of_missing_values(self)) / \
               (self.numberOfInstances * self.numberOfFeatures)

    def data_set_ratio(self):
        return self.numberOfFeatures / float(self.numberOfInstances)

    def log_data_set_ratio(self):
        return np.log(DataSetFeatureExtractor.data_set_ratio(self))

    def inverse_data_set_ratio(self):
        return float(1 / DataSetFeatureExtractor.number_of_instances(self))

    def log_inverse_data_set_ratio(self):
        return np.log(DataSetFeatureExtractor.inverse_data_set_ratio(self))

    ##################################################################################
    # Statistical meta features
    # Only use third and fourth statistical moment because it is common to standardize
    # for the other two, see Engels & Theusinger, 1998 - Using a Data Metric for
    # Preprocessing Advice for Data Mining Applications.

    def kurtosis_min(self):
        minimum = np.nanmin(self.kurts) if len(self.kurts) > 0 else 0

        return minimum if np.isfinite(minimum) else 0

    def kurtosis_max(self):
        maximum = np.nanmax(self.kurts) if len(self.kurts) > 0 else 0

        return maximum if np.isfinite(maximum) else 0

    def kurtosis_mean(self):
        mean = np.nanmean(self.kurts) if len(self.kurts) > 0 else 0

        return mean if np.isfinite(mean) else 0

    def kurtosis_std(self):
        std = np.nanstd(self.kurts) if len(self.kurts) > 0 else 0

        return std if np.isfinite(std) else 0

    def skewness_min(self):
        minimum = np.nanmin(self.skews) if len(self.skews) > 0 else 0

        return minimum if np.isfinite(minimum) else 0

    def skewness_max(self):
        maximum = np.nanmax(self.skews) if len(self.skews) > 0 else 0

        return maximum if np.isfinite(maximum) else 0

    def skewness_mean(self):
        mean = np.nanmean(self.skews) if len(self.skews) > 0 else 0

        return mean if np.isfinite(mean) else 0

    def skewness_std(self):
        std = np.nanstd(self.skews) if len(self.skews) > 0 else 0

        return std if np.isfinite(std) else 0

    def class_entropy(self):
        labels = 1 if len(self.y.shape) == 1 else self.y.shape[1]
        new_y = self.y.reshape((-1, 1)) if labels == 1 else self.y

        entropies = []
        for i in range(labels):
            occurrence_dict = defaultdict(float)

            for value in new_y[:, i]:
                occurrence_dict[value] += 1
            entropies.append(scipy.stats.entropy([occurrence_dict[key] for key in occurrence_dict], base=2))

        return np.mean(entropies)

    ################################################################################
    # Bardenet 2013 - Collaborative Hyperparameter Tuning
    # K number of classes ("number_of_classes")
    # log(d), log(number of attributes)
    # log(n/d), log(number of training instances/number of attributes)
    # p, how many principal components to keep in order to retain 95% of the
    # dataset variance
    # skewness of a dataset projected onto one principal component...
    # kurtosis of a dataset projected onto one principal component

    def pca(self):
        if not sps.issparse(self.x_clean):
            pca = PCA(copy=True)
            rs = np.random.RandomState(42)
            indices = np.arange(self.x_clean.shape[0])

            try:
                rs.shuffle(indices)
                pca.fit(self.x_clean[indices])

                return pca

            except np.linalg.LinAlgError as e:
                pass
            # self.logger.warning("Failed to compute a Principle Component Analysis")
            return None

        else:
            rs = np.random.RandomState(42)
            indices = np.arange(self.x_clean.shape[0])
            xt = self.x_clean.astype(np.float64)

            for i in range(10):
                try:
                    rs.shuffle(indices)
                    truncated_svd = TruncatedSVD(
                        n_components=xt.shape[1] - 1, random_state=i,
                        algorithm="randomized")

                    truncated_svd.fit(xt[indices])

                    return truncated_svd

                except np.linalg.LinAlgError as e:
                    pass
            # self.logger.warning("Failed to compute a Truncated SVD")
            return None

    def pca_fraction_of_components_for_95_percent_variance(self):
        pca_ = DataSetFeatureExtractor.pca(self)
        if pca_ is None:

            return np.NaN

        sum_ = 0.
        idx = 0

        while sum_ < 0.95 and idx < len(pca_.explained_variance_ratio_):
            sum_ += pca_.explained_variance_ratio_[idx]
            idx += 1

        return float(idx) / float(self.x_clean.shape[1])

    def pca_kurtosis_first_pc(self):
        pca_ = DataSetFeatureExtractor.pca(self)
        if pca_ is None:

            return np.NaN

        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(self.x_clean)
        pca_.components_ = components

        kurtosis = scipy.stats.kurtosis(transformed)

        return kurtosis[0]

    def pca_skewness_first_pc(self):
        pca_ = DataSetFeatureExtractor.pca(self)
        if pca_ is None:

            return np.NaN

        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(self.x_clean)
        pca_.components_ = components

        skewness = scipy.stats.skew(transformed)

        return skewness[0]

    ################################################################################
    # Landmarking features, computed with cross validation
    # These should be invoked with the same transformations of X and y with which
    # sklearn will be called later on
    # from Pfahringer 2000
    # Linear discriminant learner

    def landmark_lda(self):
        # TODO: modify these part
        # to not get a nan, n_split has to be at least 2 and at most the number of rows in x_clean
        n_split = 2

        if not sps.issparse(self.x_clean):
            kf = StratifiedKFold(n_splits=n_split)
            accuracy = 0.

            try:
                for train, test in kf.split(self.x_clean, self.y_clean):
                    lda = LinearDiscriminantAnalysis()
                    lda.fit(self.x_clean[train], self.y_clean[train])

                    predictions = lda.predict(self.x_clean[test])
                    accuracy += sklearn.metrics.accuracy_score(predictions, self.y_clean[test])
                return accuracy / n_split

            except scipy.linalg.LinAlgError as e:
                # self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
                return np.NaN

            except ValueError as e:
                # self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
                return np.NaN
        else:
            return np.NaN

    def landmark_naive_bayes(self):
        n_split = 2

        if not sps.issparse(self.x_clean):
            kf = StratifiedKFold(n_splits=n_split)

            accuracy = 0.
            for train, test in kf.split(self.x_clean, self.y_clean):
                nb = GaussianNB()
                nb.fit(self.x_clean[train], self.y_clean[train])

                predictions = nb.predict(self.x_clean[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, self.y_clean[test])
            return accuracy / n_split

        else:
            return np.NaN

    def landmark_decision_tree(self):
        n_split = 2

        if not sps.issparse(self.x_clean):
            kf = StratifiedKFold(n_splits=n_split)
            accuracy = 0.

            for train, test in kf.split(self.x_clean, self.y_clean):
                random_state = sklearn.utils.check_random_state(42)
                tree = DecisionTreeClassifier(random_state=random_state)

                tree.fit(self.x_clean[train], self.y_clean[train])
                predictions = tree.predict(self.x_clean[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, self.y_clean[test])
            return accuracy / n_split

        else:
            return np.NaN

    def landmark_decision_node_learner(self):
        n_split = 2

        if not sps.issparse(self.x_clean):
            kf = StratifiedKFold(n_splits=n_split)
            accuracy = 0.

            for train, test in kf.split(self.x_clean, self.y_clean):
                random_state = sklearn.utils.check_random_state(42)
                node = DecisionTreeClassifier(
                    criterion="entropy", max_depth=1, random_state=random_state,
                    min_samples_split=2, min_samples_leaf=1, max_features=None)

                node.fit(self.x_clean[train], self.y_clean[train])
                predictions = node.predict(self.x_clean[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, self.y_clean[test])
            return accuracy / n_split

        else:
            return np.NaN

    def landmark_random_node_learner(self):
        n_split = 2

        if not sps.issparse(self.x_clean):
            kf = StratifiedKFold(n_splits=n_split)
            accuracy = 0.

            for train, test in kf.split(self.x_clean, self.y_clean):
                random_state = sklearn.utils.check_random_state(42)
                node = DecisionTreeClassifier(
                    criterion="entropy", max_depth=1, random_state=random_state,
                    min_samples_split=2, min_samples_leaf=1, max_features=1)

                node.fit(self.x_clean[train], self.y_clean[train])
                predictions = node.predict(self.x_clean[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, self.y_clean[test])
            return accuracy / n_split

        else:
            return np.NaN

    # Replace the Elite 1NN with a normal 1NN, this slightly changes the
    # intuition behind this landmark, but Elite 1NN is used nowhere else...

    def landmark_1nn(self):
        n_split = 2

        kf = StratifiedKFold(n_splits=n_split)

        accuracy = 0.
        for train, test in kf.split(self.x_clean, self.y_clean):
            kNN = KNeighborsClassifier(n_neighbors=1)

            kNN.fit(self.x_clean[train], self.y_clean[train])
            predictions = kNN.predict(self.x_clean[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y_clean[test])

        return accuracy / n_split

    def class_probability_min(self):
        min_value = np.iinfo(np.int64).max

        for num_occurrences in self.occurrence_dict.values():
            if num_occurrences < min_value:
                min_value = num_occurrences

        return float(min_value) / float(self.numberOfInstances)

    def class_probability_max(self):
        max_value = -1

        for num_occurrences in self.occurrence_dict.values():
            if num_occurrences > max_value:
                max_value = num_occurrences

        return float(max_value) / float(self.numberOfInstances)

    def class_probability_mean(self):
        occurrences = np.array([occurrence for occurrence in self.occurrence_dict.values()], dtype=np.float64)

        return (occurrences / self.numberOfInstances).mean()

    def class_probability_std(self):
        occurrences = np.array([occurrence for occurrence in self.occurrence_dict.values()], dtype=np.float64)

        return (occurrences / self.numberOfInstances).std()


# Test the feature extractor
if __name__ == '__main__':
    meta_features = ['number_of_instances',
                     'log_number_of_instances',
                     'data_set_ratio',
                     'log_data_set_ratio',
                     'inverse_data_set_ratio',
                     'log_inverse_data_set_ratio',
                     'class_probability_min',
                     'class_probability_max',
                     'class_probability_mean',
                     'class_probability_std',
                     'kurtosis_min',
                     'kurtosis_max',
                     'kurtosis_mean',
                     'kurtosis_std',
                     'skewness_min',
                     'skewness_max',
                     'skewness_mean',
                     'skewness_std',
                     'class_entropy',
                     'landmark_lda',
                     'landmark_naive_bayes',
                     'landmark_decision_tree',
                     'landmark_decision_node_learner',
                     'landmark_random_node_learner',
                     'landmark_1nn',
                     'pca_fraction_of_components_for_95_percent_variance',
                     'pca_kurtosis_first_pc',
                     'pca_skewness_first_pc']

    x = np.random.rand(100, 10)
    y = np.random.randint(2, size=100)

    print DataSetFeatureExtractor(x, y).extract_features(meta_features)
