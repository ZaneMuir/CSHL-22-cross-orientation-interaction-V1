import matplotlib.pyplot as plt
import numpy as np

def plot_matrix_as_image(matrix, **kwargs):
	fig, ax = plt.subplots(1, 1)
	_plot_matrix_as_image(ax, matrix, **kwargs)
	# fig.set_size_inches(6, 6)
	# fig.set(facecolor="#00000000")
	return fig, ax

def _plot_matrix_as_image(ax, matrix, **kwargs):
	ax.imshow(matrix, vmin=-1, vmax=1, cmap='gray', **kwargs)
	ax.set_axis_off()
	ax.set_ylim((0.5, matrix.shape[0]-0.5))
	
def make_gaussian_contour(grid_size=256, blob_size=5, xc=0, yc=0, eccentricity=0, orientation=0, spatial_scale=1, step=0.01):
    """make a contour line for 2d gaussian.
    
    ## Arguments
    - grid_size: size of the image (pixel)
    - blob_size: the (y-axis) width of the 1σ gaussian contour
    - xc, yc: the location of center
    - eccentricity: the eccentricity of the gaussian shape
    - orientation: the orientation of x-axis
    - spatial_scale: linear scaler for blob_size (mostly for dev)
    - step: size in radian for each step.
    
    ## Returns
    - x: x coordinates (1d array)
    - y: y coordinates (1d array)
    """
    _center_x = (1 + grid_size) / 2
    _center_y = (1 + grid_size) / 2
    # eccentricity = 1/xy_ratio
    
    _θ = np.linspace(-step, 2 * np.pi, int(np.floor(2*np.pi/step))+1)
    # _r = blob_size * (1-eccentricity**2)/(1+eccentricity * np.cos(_θ))
    
    _a = blob_size * spatial_scale #TODO: why inverse?
    _b = np.sqrt((1 - eccentricity**2) * _a ** 2)
    
    _x = np.cos(_θ) * _a
    _y = np.sin(_θ) * _b
    
    _ori_rad = np.deg2rad(orientation)
    _ori_Q = np.array([
        [np.cos(_ori_rad), -np.sin(_ori_rad)],
        [np.sin(_ori_rad), np.cos(_ori_rad)]
    ])
    # print(_ori_Q)
    
    _contour = _ori_Q @ np.vstack((_x, _y)) * spatial_scale + np.array([[xc+_center_x], [yc+_center_y]])
    # print(np.vstack((_x, _y)).shape)
    
    return _contour[0], _contour[1] 
    
def plot_ellipse_contour(ax, plot_param={}, **kwargs):
	ax.plot(*make_gaussian_contour(**kwargs), **plot_param)
	