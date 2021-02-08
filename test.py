import torch
import numpy as np
from imblearn.over_sampling import RandomOverSampler


class Data(object):
    def __init__(self, x) -> None:
        super().__init__()
        self.x = x
    

"""Base class for sampling"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import check_classification_targets


from imblearn.utils import check_sampling_strategy, check_target_type
from imblearn.utils._validation import ArraysTransformer
from imblearn.utils._validation import _deprecate_positional_args


from collections import Counter

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_target_type
from imblearn.utils import Substitution
from imblearn.utils._docstring import _random_state_docstring
# from imblearn.utils._validation import _deprecate_positional_args


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class RandomOverSampler(BaseOverSampler):
    """Class to perform random over-sampling.

    Object to over-sample the minority class(es) by picking samples at random
    with replacement.

    Read more in the :ref:`User Guide <random_over_sampler>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    Attributes
    ----------
    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    See Also
    --------
    ROSE : Random Over-Sampling Examples.

    BorderlineSMOTE : Over-sample using the bordeline-SMOTE variant.

    SMOTE : Over-sample using SMOTE.

    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import \
RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> ros = RandomOverSampler(random_state=42)
    >>> X_res, y_res = ros.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})
    """

    @_deprecate_positional_args
    def __init__(self, *, sampling_strategy="auto", random_state=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state

    def _check_X_y(self, X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = self._validate_data(
            X, y, reset=True, accept_sparse=["csr", "csc"], dtype=None,
            force_all_finite=False,
        )
        return X, y, binarize_y

    def _fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)
        target_stats = Counter(y)

        sample_indices = range(X.shape[0])

        for class_sample, num_samples in self.sampling_strategy_.items():
            target_class_indices = np.flatnonzero(y == class_sample)
            indices = random_state.randint(
                low=0, high=target_stats[class_sample], size=num_samples
            )

            sample_indices = np.append(
                sample_indices, target_class_indices[indices]
            )
        self.sample_indices_ = np.array(sample_indices)

        return (
            _safe_indexing(X, sample_indices),
            _safe_indexing(y, sample_indices),
        )

    def _more_tags(self):
        return {
            "X_types": ["2darray", "string"],
            "sample_indices": True,
            "allow_nan": True,
        }



class SamplerMixin(BaseEstimator, metaclass=ABCMeta):
    """Mixin class for samplers with abstract method.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _estimator_type = "sampler"

    def fit(self, X, y):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples,)
            Target array.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        X, y, _ = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )
        return self

    def fit_resample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe, sparse matrix} of shape \
                (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, dataframe, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new,)
            The corresponding label of `X_resampled`.
        """
        check_classification_targets(y)
        arrays_transformer = ArraysTransformer(X, y)
        X, y, binarize_y = self._check_X_y(X, y)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type
        )

        output = self._fit_resample(X, y)

        y_ = (label_binarize(output[1], np.unique(y))
              if binarize_y else output[1])

        X_, y_ = arrays_transformer.transform(output[0], y_)
        return (X_, y_) if len(output) == 2 else (X_, y_, output[2])

    #  define an alias for back-compatibility
    fit_sample = fit_resample

    @abstractmethod
    def _fit_resample(self, X, y):
        """Base method defined in each sampler to defined the sampling
        strategy.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray of shape (n_samples_new,)
            The corresponding label of `X_resampled`.

        """
        pass


class BaseSampler(SamplerMixin):
    """Base class for sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, sampling_strategy="auto"):
        self.sampling_strategy = sampling_strategy

    def _check_X_y(self, X, y, accept_sparse=None):
        if accept_sparse is None:
            accept_sparse = ["csr", "csc"]
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = self._validate_data(
            X, y, reset=True, accept_sparse=accept_sparse
        )
        return X, y, binarize_y




ros = RandomOverSampler(random_state=0)


# X = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18],[19,20,21],[0.1,0.2,0.3],[0.4,0.5,0.6]])
# y = torch.tensor([0,0,0,0,0,0,0,1,1])

X = [np.array([1,2,3]), np.array([4,5,6]), np.array([7,8,9]), np.array([10,11,12]), np.array([0.1,0.2,0.3])]
y = [0,0,0,0,1]

# X2 = [Data(k) for k in X]


X_resampled, y_resampled = ros.fit_resample(X, y)


print("Done!")