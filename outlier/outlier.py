
__all__ = [
    'remove_outlier'
]


def remove_outlier(df, col, partitionby=None, kernel='both', n_std=2, eps=1e-5):
    """
    :param partitionby: remove outlier in each partition
    :param kernel: kernel options: ["std", "quartile", "both"]
    :type df: pandas.DataFrame
    :type col: str
    :type partitionby: list
    :type kernel: str
    :type n_std: int
    :type eps: float
    :return: pandas.DataFrame
    """
    import numpy as np
    if partitionby is None:
        return _remove_outlier(df, col, kernel, n_std, eps)
    else:
        partitions = df[partitionby].drop_duplicates().values
        result_df = None
        for v in partitions:
            # select each partition
            sub_df = df[np.all((df[partitionby].values == v), axis=1)]
            sub_df = _remove_outlier(sub_df, col, kernel, n_std, eps)
            if result_df is None:
                result_df = sub_df
            else:
                result_df = result_df.append(sub_df)
        return result_df


def _remove_outlier(df, col, kernel='both', n_std=2, eps=1e-5):
    if kernel == 'std':
        return _remove_outlier_using_std(df, col, n_std, eps)
    elif kernel == 'quartile':
        return _remove_outlier_using_quartile(df, col, eps)
    elif kernel == 'both':
        return _remove_outlier_using_std_and_quartile(df, col, n_std, eps)
    else:
        raise BaseException('kernel options: ["std", "quartile", "both"]')


def _remove_outlier_using_std_and_quartile(df, col, n_std=2, eps=1e-5):
    mean, std = df[col].mean(), df[col].std(ddof=0)
    QL, QU = df[col].quantile([0.25, 0.75])
    lower, upper = max(mean - n_std*std, 2.5 * QL - 1.5 * QU), min(mean + n_std*std, 2.5 * QU - 1.5 * QL)
    return df[(df[col] > lower - eps) & (df[col] < upper + eps)]


def _remove_outlier_using_std(df, col, n_std=2, eps=1e-5):
    mean, std = df[col].mean(), df[col].std(ddof=0)
    lower, upper = mean - n_std*std, mean + n_std*std
    return df[(df[col] > lower - eps) & (df[col] < upper + eps)]


def _remove_outlier_using_quartile(df, col, eps=1e-5):
    QL, QU = df[col].quantile([0.25, 0.75])
    lower, upper = 2.5*QL - 1.5*QU, 2.5*QU - 1.5*QL
    return df[(df[col] > lower - eps) & (df[col] < upper + eps)]

