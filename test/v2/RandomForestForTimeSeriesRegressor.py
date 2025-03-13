from logging import warn
from warnings import catch_warnings, simplefilter

import numpy as np
from joblib import delayed
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import issparse
from sklearn.ensemble._forest import _get_n_samples_bootstrap, _parallel_build_trees, _generate_sample_indices
from sklearn.exceptions import DataConversionWarning
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import compute_sample_weight
from sklearn.utils._random import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.parallel import Parallel
from sklearn.utils.validation import _check_sample_weight
MAX_INT = np.iinfo(np.int32).max
BLOCK_TYPES = ["non-overlapping", "moving-block", "circular-block", "non-overlapping-alter", "non-overlapping-single"]
class RandomForestForTimeSeriesRegressor(RandomForestRegressor):

    # Llamo al constructor de RandomForestRegressor
    # Necesario porque quiero una variable con el tamaño de bloques
    # block_size

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        block_size=1,
        block_type="non-overlapping"
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.block_size = block_size
        self.block_type = block_type


    # Función copypasteada desde la clase sklearn.ensemble._forest
    def fit(self, X, y, sample_weight=None):

        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")

        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            accept_sparse="csc",
            dtype=DTYPE,
            force_all_finite=False,
        )
        # _compute_missing_values_in_feature_mask checks if X has missing values and
        # will raise an error if the underlying tree base estimator can't handle missing
        # values. Only the criterion is required to determine if the tree supports
        # missing values.
        estimator = type(self.estimator)(criterion=self.criterion)
        missing_values_in_feature_mask = (
            estimator._compute_missing_values_in_feature_mask(
                X, estimator_name=self.__class__.__name__
            )
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                (
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples,), for example using ravel()."
                ),
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self._n_samples, self.n_outputs_ = y.shape

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # TODO: Analyse

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            # La función devuelve el valor de max_samples, cuando n_samples es menor
            # if max_samples > n_samples:
            #     msg = "`max_samples` must be <= n_samples={} but got value {}"
            #     raise ValueError(msg.format(n_samples, max_samples))
            # return max_samples
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._n_samples_bootstrap = n_samples_bootstrap

        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available if bootstrap=True")
        # inicializa random_state
        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))
            # Acá instancia los árboles sin entrenar
            trees = [
                self._make_estimator(append=False, random_state=random_state)
                for i in range(n_more_estimators)
            ]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )( # La función que se llama acá es la que hace el subsampling
                delayed(_parallel_build_trees_with_blocks)( # Cambiada la llamada a la función
                    t,
                    self.bootstrap,
                    X,
                    y,
                    None,
                    i,
                    len(trees),
                    verbose=self.verbose,
                    class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap,
                    missing_values_in_feature_mask=missing_values_in_feature_mask,
                    block_type=self.block_type,
                    block_size=self.block_size,
                )
                for i, t in enumerate(trees)
            )

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score and (
                n_more_estimators > 0 or not hasattr(self, "oob_score_")
        ):
            y_type = type_of_target(y)
            if y_type == "unknown" or (
                    self._estimator_type == "classifier"
                    and y_type == "multiclass-multioutput"
            ):
                # FIXME: we could consider to support multiclass-multioutput if
                # we introduce or reuse a constructor parameter (e.g.
                # oob_score) allowing our user to pass a callable defining the
                # scoring strategy on OOB sample.
                raise ValueError(
                    "The type of target cannot be used to compute OOB "
                    f"estimates. Got {y_type} while only the following are "
                    "supported: continuous, continuous-multioutput, binary, "
                    "multiclass, multilabel-indicator."
                )

            if callable(self.oob_score):
                self._set_oob_score_and_attributes(
                    X, y, scoring_function=self.oob_score
                )
            else:
                self._set_oob_score_and_attributes(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self
# Función copipasteada desde la clase sklearn.ensemble._forest
# Armar bloques acá
def _parallel_build_trees_with_blocks(
    tree,
    bootstrap,
    X,
    y,
    sample_weight,
    tree_idx,
    n_trees,
    verbose=0,
    class_weight=None,
    n_samples_bootstrap=None,
    missing_values_in_feature_mask=None,
    block_size=1,
    block_type=BLOCK_TYPES[0]
):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    n_samples = X.shape[0]
    if sample_weight is None:
        curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
    else:
        curr_sample_weight = sample_weight.copy()

    # Acá le da pesos a los datos de forma aleatoria
    # Se puede modificar acá así se le da peso a uno y a los N siguientes...
    # indices = _generate_sample_indices(
    #     tree.random_state, n_samples, n_samples_bootstrap
    # )
    #sample_counts = np.bincount(indices, minlength=n_samples)
    #sample_counts = [peso índice 0, peso índice 1, ..., peso índice N]
    #sample_counts = [0] * n_samples
    indices = []
    block_count = n_samples // block_size
    if block_type == BLOCK_TYPES[0]:
        #Non overlapping
        # pivot = np.random.randint(n_samples - block_size)
        # for i in range(block_size):
        #     sample_counts[(pivot + i) % n_samples] = int(n_samples // block_size)
        #     indices.append((pivot + i) % n_samples)
        # curr_sample_weight *= sample_counts
        indices = set(indices)
        for i in range(block_count):
            indices0 = generate_block_non_overlapping(block_size, n_samples)
            # for idx in indices0:
            #     #sample_counts[idx] += 1
            #     if idx not in indices:
            #         indices.append(idx)
            indices = indices.union(indices0)
        indices = list(indices)
    else:

        # Genero bloques con pivotes aleatorios y los junto.
        if block_type == BLOCK_TYPES[1]:

            for i in range(block_count):
                indices0 = generate_moving_block(block_size, n_samples)
                for idx in indices0:
                    #sample_counts[idx] += 1
                    indices.append(idx)

        if block_type == BLOCK_TYPES[2]:

            for i in range(block_count):
                indices0 = generate_circular_block(block_size, n_samples)
                for idx in indices0:
                    #sample_counts[idx] += 1
                    indices.append(idx)

    sample_counts = np.bincount(indices, minlength=n_samples)
    curr_sample_weight *= sample_counts

    #########################################################################
    # No sé que hace esto pero desordena los pesos, quitar si es posible
    if class_weight == "subsample":
        with catch_warnings():
            simplefilter("ignore", DeprecationWarning)
            curr_sample_weight *= compute_sample_weight("auto", y, indices=indices)
    elif class_weight == "balanced_subsample":
        curr_sample_weight *= compute_sample_weight("balanced", y, indices=indices)
    ###########################################################################
    tree._fit(
        X,
        y,
        sample_weight=curr_sample_weight,
        check_input=False,
        missing_values_in_feature_mask=missing_values_in_feature_mask,
    )

    return tree

def generate_moving_block(block_size, n_samples):
    k = n_samples // block_size
    offset = np.random.randint(k) * block_size
    #pivot = np.random.randint(n_samples - block_size)
    indices = []
    for i in range(block_size):
        indices.append((offset + i))
    return indices

def generate_circular_block(block_size, n_samples):
    k = n_samples // block_size
    n = 2
    offset = np.random.randint(k * n) * block_size
    #pivot = np.random.randint(n_samples)
    indices = []
    for i in range(block_size):
        indices.append((offset + i) % n_samples)
    return indices

def generate_block_non_overlapping(block_size, n_samples):
    pivot = np.random.randint(n_samples - block_size)
    indices = []
    for i in range(block_size):
        indices.append((pivot + i))
    return indices