from .stimulus import *
from .visualization import *
from .linear_filter import *
from .nonlinearity import *

# def contrast_response_curve(order=1, max_rate=11.3, contrast=0, background_rate=0, c50=50):
#     return max_rate * (contrast ** order) / (contrast ** order + c50 ** order) + background_rate

from .point_model import *
from .metrics import *