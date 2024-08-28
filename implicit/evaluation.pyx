# distutils: language = c++

import cython
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from tqdm.auto import tqdm

from libc.math cimport fmin
from libcpp.unordered_set cimport unordered_set

from .utils import check_random_state


def train_test_split(ratings, train_percentage=0.8, random_state=None):
    """ Randomly splits the ratings matrix into two matrices for training/testing.

    Parameters
    ----------
    ratings : coo_matrix
        A sparse matrix to split
    train_percentage : float
        What percentage of ratings should be used for training
    random_state : int, None or RandomState
        The existing RandomState. If None, or an int, will be used
        to seed a new numpy RandomState.
    Returns
    -------
    (train, test) : csr_matrix, csr_matrix
        A tuple of csr_matrices for training/testing """

    ratings = ratings.tocoo()
    random_state = check_random_state(random_state)
    random_index = random_state.random(len(ratings.data))
    train_index = random_index < train_percentage
    test_index = random_index >= train_percentage

    train = csr_matrix((ratings.data[train_index],
                        (ratings.row[train_index], ratings.col[train_index])),
                       shape=ratings.shape, dtype=ratings.dtype)

    test = csr_matrix((ratings.data[test_index],
                       (ratings.row[test_index], ratings.col[test_index])),
                      shape=ratings.shape, dtype=ratings.dtype)

    test.data[test.data < 0] = 0
    test.eliminate_zeros()

    return train, test


cdef _choose(rng, int n, float frac):
    """Given a range of numbers, select *approximately* 'frac' of them _without_
    replacement.

    Parameters
    ----------
    rng : int, None or RandomState
        The existing RandomState. If None, or an int, will be used
        to seed a new numpy RandomState.
    n: int
        The upper bound on the range to sample from. Will draw from range(0 -> n).
    frac: float
        The fraction of the total range to be sampled. Must be in interval (0 -> 1).

    Returns
    -------
    ndarray
        An array of randomly sampled integers in the range (0 -> n).

    """

    size = max(1, int(n * frac))
    arr = rng.choice(n, size=size, replace=False)
    return arr

cdef _take_tails(arr, int n, return_complement=False, shuffled=False, included_elements=None):
    """
    Modified _take_tails to ensure that only specific elements are included in the tails.

    Parameters
    ----------
    included_elements: list or None
        If not None, only these elements will be considered for tails.
    """
    if included_elements is not None:
        # Filter the array to include only specified elements
        mask = np.isin(arr, included_elements)
        arr = arr[mask]
    
    idx = arr.argsort()
    sorted_arr = arr[idx]

    end = np.bincount(sorted_arr).cumsum() - 1
    start = end - n
    ranges = np.linspace(start, end, num=n + 1, dtype=int)[1:]

    if shuffled:
        shuffled_idx = (sorted_arr + np.random.random(arr.shape)).argsort()
        tails = shuffled_idx[np.ravel(ranges, order="f")]
    else:
        tails = np.ravel(ranges, order="f")

    heads = np.setdiff1d(idx, tails)

    if return_complement:
        return idx[tails], idx[heads]
    else:
        return idx[tails]
    
cpdef leave_k_out_split(
    ratings, int K=1, float train_only_size=0.0, random_state=None, test_items=None
):
    """
    Implements the 'leave-k-out' split protocol with the ability to exclude items
    not specified in `test_items` from the test set.

    Parameters
    ----------
    test_items : list or None
        A list of item IDs that must be considered for the test set. All other items
        will be excluded.
    """

    if K < 1:
        raise ValueError("The 'K' must be >= 1.")
    if not 0.0 <= train_only_size < 1.0:
        raise ValueError("The 'train_only_size' must be in the range (0.0 <= x < 1.0).")

    ratings = ratings.tocoo()
    random_state = check_random_state(random_state)

    users = ratings.row
    items = ratings.col
    data = ratings.data

    # If test_items is provided, filter users and items based on this
    if test_items is not None:
        mask = np.isin(items, test_items)
        users = users[mask]
        items = items[mask]
        data = data[mask]

    unique_users, counts = np.unique(users, return_counts=True)

    # Diagnostic print statement
    print(f"Total unique users before any filtering: {len(unique_users)}")

    # Now the candidate_mask is based only on the filtered items
    candidate_mask = counts > K + 1

    # Diagnostic print statement
    print(f"Unique users after filtering by K + 1: {len(unique_users[candidate_mask])}")

    if train_only_size > 0.0:
        train_only_mask = ~np.isin(
            unique_users, _choose(random_state, len(unique_users), train_only_size)
        )
        candidate_mask = train_only_mask & candidate_mask

    unique_candidate_users = unique_users[candidate_mask]
    full_candidate_mask = np.isin(users, unique_candidate_users)

    # Diagnostic print statement
    print(f"Unique users after applying included_elements: {len(unique_candidate_users)}")

    candidate_users = users[full_candidate_mask]
    candidate_items = items[full_candidate_mask]
    candidate_data = data[full_candidate_mask]

    test_idx, train_idx = _take_tails(
        candidate_users, K, shuffled=True, return_complement=True, included_elements=test_items
    )

    #train_idx = np.setdiff1d(np.arange(len(candidate_users), dtype=int), test_idx)

    test_users = candidate_users[test_idx]
    test_items_filtered = candidate_items[test_idx]
    test_data = candidate_data[test_idx]
    test_mat = csr_matrix(
        (test_data, (test_users, test_items_filtered)), shape=ratings.shape, dtype=ratings.dtype
    )

    train_users = np.r_[users[~full_candidate_mask], candidate_users[train_idx]]
    train_items = np.r_[items[~full_candidate_mask], candidate_items[train_idx]]
    train_data = np.r_[data[~full_candidate_mask], candidate_data[train_idx]]
    train_mat = csr_matrix(
        (train_data, (train_users, train_items)),
        shape=ratings.shape,
        dtype=ratings.dtype,
    )

    train_users = np.unique(train_users)  # Count unique users in the train set
    test_users = np.unique(test_users)    # Count unique users in the test set
    
    # Diagnostic print statement
    print(f"Number of unique users in train set after _take_tails: {len(np.unique(train_users))}")
    print(f"Number of unique users in test set after _take_tails: {len(np.unique(test_users))}")

    return train_mat, test_mat


@cython.boundscheck(False)
def precision_at_k(model, train_user_items, test_user_items, int K=10,
                   show_progress=True, int num_threads=1, included_elements=None):
    """ Calculates P@K for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used
            in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to
        test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated p@k
    """
    return ranking_metrics_at_k(
        model, train_user_items, test_user_items, K, show_progress, num_threads, included_elements
    )['precision']


@cython.boundscheck(False)
def mean_average_precision_at_k(model, train_user_items, test_user_items, int K=10,
                                show_progress=True, int num_threads=1, included_elements=None):
    """ Calculates MAP@K for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated MAP@k
    """
    return ranking_metrics_at_k(
        model, train_user_items, test_user_items, K, show_progress, num_threads, included_elements
    )['map']


@cython.boundscheck(False)
def ndcg_at_k(model, train_user_items, test_user_items, int K=10,
              show_progress=True, int num_threads=1, included_elements=None):
    """ Calculates ndcg@K for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated ndcg@k
    """
    return ranking_metrics_at_k(
        model, train_user_items, test_user_items, K, show_progress, num_threads, included_elements
    )['ndcg']


@cython.boundscheck(False)
def AUC_at_k(model, train_user_items, test_user_items, int K=10,
             show_progress=True, int num_threads=1, included_elements=None):
    """ Calculate limited AUC for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated AUC@k
    """
    return ranking_metrics_at_k(
        model, train_user_items, test_user_items, K, show_progress, num_threads, included_elements
    )['auc']


@cython.boundscheck(False)
def ranking_metrics_at_k(
    model, 
    train_user_items, 
    test_user_items, 
    int K=10,
    show_progress=True, 
    int num_threads=1, 
    included_elements=None
):
    """ Calculates ranking metrics for a given trained model, filtering on included_elements.

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used
            in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to
        test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.
    included_elements : array_like, optional
        List of item ids to be considered in the evaluation. Only these items will
        be used for calculating the metrics.

    Returns
    -------
    dict
        A dictionary containing the calculated metrics: precision, map, ndcg, auc
    """
    if not isinstance(train_user_items, csr_matrix):
        train_user_items = train_user_items.tocsr()

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()

    cdef int users = test_user_items.shape[0], items = test_user_items.shape[1]
    cdef int u, i, batch_idx
    # precision
    cdef double relevant = 0, pr_div = 0, total = 0
    # map
    cdef double mean_ap = 0, ap = 0
    # ndcg
    cdef double[:] cg = (1.0 / np.log2(np.arange(2, K + 2)))
    cdef double[:] cg_sum = np.cumsum(cg)
    cdef double ndcg = 0, idcg
    # auc
    cdef double mean_auc = 0, auc, hit, miss, num_pos_items, num_neg_items

    cdef int[:] test_indptr = test_user_items.indptr
    cdef int[:] test_indices = test_user_items.indices

    cdef int[:, :] ids
    cdef int[:] batch

    cdef unordered_set[int] likes

    cdef set included_elements_set
    if included_elements is not None:
        included_elements_set = set(included_elements)

    batch_size = 1000
    start_idx = 0

    # get an array of userids that have at least one item in the test set
    to_generate = np.arange(users, dtype="int32")
    to_generate = to_generate[np.ediff1d(test_user_items.indptr) > 0]

    progress = tqdm(total=len(to_generate), disable=not show_progress)

    while start_idx < len(to_generate):
        batch = to_generate[start_idx: start_idx + batch_size]

        # Recommend items using the filter_items parameter
        ids, _ = model.recommend(
            batch, 
            train_user_items[batch], 
            N=K, 
            filter_items=included_elements  # Filter out items not in included_elements
        )
        start_idx += batch_size

        with nogil:
            for batch_idx in range(len(batch)):
                u = batch[batch_idx]
                likes.clear()

                # Temporarily acquire the GIL to work with Python objects
                with gil:
                    for i in range(test_indptr[u], test_indptr[u + 1]):
                        # The critical section that requires Python object handling.
                        if included_elements_set is None or test_indices[i] in included_elements_set:
                            likes.insert(test_indices[i])

                pr_div += fmin(K, likes.size())
                ap = 0
                hit = 0
                miss = 0
                auc = 0
                idcg = cg_sum[min(K, likes.size()) - 1]
                num_pos_items = likes.size()
                num_neg_items = items - num_pos_items

                for i in range(K):
                    if likes.find(ids[batch_idx, i]) != likes.end():
                        relevant += 1
                        hit += 1
                        ap += hit / (i + 1)
                        ndcg += cg[i] / idcg
                    else:
                        miss += 1
                        auc += hit
                auc += ((hit + num_pos_items) / 2.0) * (num_neg_items - miss)
                mean_ap += ap / fmin(K, likes.size())
                mean_auc += auc / (num_pos_items * num_neg_items)
                total += 1

        progress.update(len(batch))

    progress.close()
    return {
        "precision": relevant / pr_div,
        "map": mean_ap / total,
        "ndcg": ndcg / total,
        "auc": mean_auc / total
    }
