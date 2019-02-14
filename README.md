# Usage

```python
def remove_outlier(df, col, partitionby=None, kernel='both', n_std=2, eps=1e-5):
    """
    :param col: label column
    :param partitionby: remove outliers in each partition
    :param kernel: kernel options: ["std", "quartile", "both"]
    :type df: pandas.DataFrame
    :type col: str
    :type partitionby: list
    :type kernel: str
    :type n_std: int
    :type eps: float
    :return: pandas.DataFrame
    """
```