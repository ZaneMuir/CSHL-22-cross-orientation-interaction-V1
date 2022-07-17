import numpy as np

def make_grating_image(grid_size=256, orientation=0, contrast=1.0, spatial_frequency=10, phase=0, style=np.sin):
    """create a grating image. Origin at bottom left corner!
    
    - grid_size: size of the image (pixel)
    - orientation: orientation of the grating, right is 0 degree (degree).
    - contrast: contrast gain of the grating
    - spatial_frequency: pixel per cycle
    - phase: phase offset in rad (rad)
    - style: waveform style (rad->contrast)
    
    """
    _pixel_axis = np.linspace(1, grid_size, grid_size)
    _pixel_dist_X = _pixel_axis.repeat(grid_size).reshape((grid_size, grid_size)).T.reshape((-1,)) - _pixel_axis.mean()
    _pixel_dist_Y = _pixel_axis.repeat(grid_size).reshape((grid_size, grid_size)).reshape((-1,)) - _pixel_axis.mean()
    _pixel_dist = np.vstack((_pixel_dist_X, _pixel_dist_Y))
    
    _ori_rad = np.deg2rad(orientation)
    _ori_Q = np.array([ # counterclockwise rotation transformation matrix
        [np.cos(_ori_rad), np.sin(_ori_rad)]])
    _pixel_dist_rotated = _ori_Q @ _pixel_dist
    
    _image_ϕ = _pixel_dist_rotated / spatial_frequency * 2 * np.pi + phase
    _image = contrast * style(_image_ϕ).reshape(grid_size, grid_size)
    
    return _image

def make_plaid_image(grid_size=256, orientation=0, contrast=1.0, phase=0, Δphase=0, *args, **kwargs):
    """create a plaid image.
    
    - grid_size: size of the image (pixel)
    - orientation: orientation of the plaid, right is 0 degree (degree).
    - contrast: joint contrast gain of the grating
    - *args, **kwargs: refer to `make_grating_image` for other arguments.
    """
    _base = make_grating_image(grid_size, orientation-45, contrast=contrast/2, phase=phase, **kwargs)
    _orthogonal = make_grating_image(grid_size, orientation+45, contrast=contrast/2, phase=phase+Δphase, **kwargs)
    return _base + _orthogonal

def make_hyperplaid_image(grid_size=256, ori1=0, ori2=0, contrast=1.0, phase1=0, phase2=0, **kwargs):
	_ori1 = make_grating_image(grid_size, ori1, contrast=contrast/2, phase=phase1, **kwargs)
	_ori2 = make_grating_image(grid_size, ori2, contrast=contrast/2, phase=phase2, **kwargs)
	return _ori1 + _ori2