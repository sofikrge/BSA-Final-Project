import numpy as np

def find_outliers(data, threshold=1.5):
    """
    Returns the indices of outliers in 'data' using the boxplot rule:
    values outside [Q1 - threshold*IQR, Q3 + threshold*IQR].
    """
    if len(data) == 0:
        return []
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
    return outlier_indices