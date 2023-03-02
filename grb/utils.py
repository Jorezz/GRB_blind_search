import numpy as np

def Chi2_polyval(x, y, param):
    """
    Returns Chi square functional for np.polyfit approximation
    """
    approximation = np.polyval(param,x)
    squared_error = np.square((np.asarray(y) - approximation))/approximation

    return np.sum(squared_error)/(len(x)-len(param))