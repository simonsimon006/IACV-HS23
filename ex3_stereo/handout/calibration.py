import numpy as np


def compute_mx_my(calib_dict):
    """
    Given a calibration dictionary, compute mx and my (in units of [px/mm]).
    
    mx -> Number of pixels per millimeter in x direction (ie width)
    my -> Number of pixels per millimeter in y direction (ie height)
    """
    
    mx = calib_dict["width"] / calib_dict["aperture_w"]
    my = calib_dict["height"] / calib_dict["aperture_h"]

    return mx, my


def estimate_f_b(calib_dict, calib_points, n_points=None):
    """
    Estimate focal lenght f and baseline b from provided calibration points.

    Note:
    In real life multiple points are useful for calibration - in case there are erroneous points.
    Here, this is not the case. It's OK to use a single point to estimate f, b.
    
    Args:
        calib_dict (dict)           ... Incomplete calibaration dictionary
        calib_points (pd.DataFrame) ... Calibration points provided with data. (Units are given in [mm])
        n_points (int)              ... Number of points used for estimation
        
    Returns:
        f   ... Focal lenght [mm]
        b   ... Baseline [mm]
    """
    
    if n_points is not None:
        calib_points = calib_points.head(n_points)
    else: 
        n_points = len(calib_points)
    
    uleft0 = calib_points["ul [px]"][0]
    uright0 = calib_points["ur [px]"][0]

    ox = calib_dict["o_x"]
    mx = calib_dict["mx"]
    x = calib_points["X [mm]"][0]
    z = calib_points["Z [mm]"][0]
    
    ratio_X_Z = x / z
    
    f = (uleft0 - ox) / (ratio_X_Z * mx)
    b = z*(ox-uright0) / (f*mx) + x
    
    return f, b