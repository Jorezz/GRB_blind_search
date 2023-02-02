import numpy as np

def Chi2_polyval(x, y, y_err, param):
    """
    Returns Chi square functional for np.polyfit approximation
    """
    approximation = np.polyval(param,x)
    squared_error = (np.asarray(y) - approximation)/np.square(np.asarray(y_err))

    return np.sum(squared_error)/(len(x)-len(param))