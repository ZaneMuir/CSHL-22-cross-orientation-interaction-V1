import numpy as np

def make_LGN_linear_filter(grid_size=256, blob_size=5, xc=0, yc=0, eccentricity=0, orientation=0, contrast=1, spatial_scale=1, norm=True):
    """make the linear filter of the LGN subunit, with compatible size/shape with the grating/plaid image.
    
    ## Arguments
    - grid_size: size of the image (pixel)
    - blob_size: the (y-axis) width of the 1σ gaussian contour
    - xc, yc: the location of center
    - eccentricity: the eccentricity of the gaussian shape
    - orientation: the orientation of x-axis
    - contrast: contrast scaler
    - spatial_scale: linear scaler for blob_size (mostly for dev)
    - norm: normalize the gaussian distribution to have 1 as the peak, so the contrast would be set accordingly
    
    ## Returns
    - linear filter in size of [grid_size x grid_size]
    """
    _pixel_axis = np.linspace(1, grid_size, grid_size)
    _pixel_dist_X = _pixel_axis.repeat(grid_size).reshape((grid_size, grid_size)).T.reshape((-1,)) - _pixel_axis.mean() - xc
    _pixel_dist_Y = _pixel_axis.repeat(grid_size).reshape((grid_size, grid_size)).reshape((-1,)) - _pixel_axis.mean() - yc
    _pixel_dist = np.vstack((_pixel_dist_X, _pixel_dist_Y))
    
    _ori_rad = np.deg2rad(orientation)
    _ori_Q = np.array([ # counterclockwise rotation transformation matrix
        [np.cos(_ori_rad), np.sin(_ori_rad)],
        [-np.sin(_ori_rad), np.cos(_ori_rad)]
    ])
    _pixel_dist_rotated = _ori_Q @ _pixel_dist * spatial_scale
    
    _a = 1/blob_size * spatial_scale #TODO: why inverse?
    _b = np.sqrt((1 - eccentricity**2) * _a ** 2)
    _image = np.exp(-((_pixel_dist_rotated[0])**2 / 2*_b**2 + (_pixel_dist_rotated[1])**2 / 2*_a**2))
    if norm:
        _image = _image / np.max(_image)
    _image = _image * contrast
    return _image.reshape(grid_size, grid_size)