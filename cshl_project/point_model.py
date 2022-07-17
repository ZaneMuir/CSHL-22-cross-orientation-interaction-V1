import numpy as np
from .nonlinearity import relu

class LGNnode(object):
    """LN model for LGN subunits
    
    This class will calculate the response to sinusoidal grating and plaid given
    by the stimulus orientation (in rad), phase (in rad), contrast ([0, 1]), etc.
    Instead of spatial convolution, the LGN subunit will be represented as a point.
    
    Parameters:
    - `center`: the center location [default: np.array([0, 0])]
    - `nonlinear`: the nonlinear function [default: relu]
    - `Rmax`: the maximum firing rate [default: 11.3]
    - `C50`: contrast level for 50% response [default: 0.5]
    - `Rbase`: background response [default: 0]
    
    Public Methods:
    - `get_response_grating`
    - `get_response_plaid`
    
    Private Methods:
    - `_get_contrast_response`
    - `_get_sinusoidal`
    - `_get_sinusoidal_plaid`
    
    """
    def __init__(self, center=(0, 0), nl=relu, Rmax=11.3, C50=0.5, Rbase=0):
        """Constructor of `LGNnode` class
        
        Keyword Arguments:
        - center: `tuple{real, real}` location of this subunit [default: (0, 0), arbitrary unit (e.g. pixel)]
        - nl: `function` nonlinearity function [default: relu]
        - Rmax: [default: 11.3] refer to `LGNnode._get_contrast_response` for full documentation.
        - C50: [default: 0.5] refer to `LGNnode._get_contrast_response` for full documentation.
        - Rbase: [default: 0] refer to `LGNnode._get_contrast_response` for full documentation.
        """
        super(LGNnode, self).__init__()
        
        self.center = np.array(center).reshape((2, 1))
        self.nonlinear = nl
        
        self.Rmax = Rmax
        self.C50 = C50
        self.Rbase = Rbase
        
    def _get_contrast_response(self, contrast, order=1):
        """Contrast response function
        
        Nonlinear map from contrast level to response (e.g., firing rate).
        
        .. math::
            R(c) = R_{\\text{max}} \\frac{c^\\alpha}{c^\\alpha + c_{50}^\\alpha} + R_{\\text{base}}
            
        Arguments:
        - contrast: `real` contrast level ranging from 0 to 1
        
        Keyword Arguments:
        - order: `real` the power of the function [default: 1]
        
        Inherited Arguments:
        - self.Rmax: the maximum response [default: 11.3]
        - self.C50: the contrast level for half of the maximum response [default: 0.5]
        - self.Rbase: the spontaneous/background response [default: 0]
        """
        return self.Rmax * contrast ** order / (contrast ** order + self.C50 ** order) + self.Rbase
    
    def _get_sinusoidal(self, ori=0, contrast=0.48, phase=0, sf=50):
        """return the luminance and contrast level given grating parameters.
        Refer to `LGNnode.get_response_grating` for full documentation.
        """
        _dist_ϕ = np.array([[np.cos(ori), np.sin(ori)]]) @ self.center / sf * 2 * np.pi
        _luminance = np.sin(_dist_ϕ[0][0] + phase)
        return _luminance * 2, contrast
    
    #XXX: check
    # def _get_sinusoidal_plaid(self, ori1=0, ori2=0, contrast1=0.48, contrast2=0.48, phase=0, Δphase=0, sf=50):
    #     _l1, _c1 = self._get_sinusoidal(ori1, contrast1, phase, sf)
    #     _l2, _c2 = self._get_sinusoidal(ori2, contrast2, phase-Δphase, sf)
    #     return (_l1+_l2) / 2, (_c1+_c2)
    
    def _get_sinusoidal_plaid(self, ori1=0, ori2=0, contrast1=0.48, contrast2=0.48, phase=0, Δphase=0, sf=50):
        """return the luminance and contrast level given plaid parameters.
        Refer to `LGNnode.get_response_plaid` for full documentation.
        """
        _dist_ϕ_1 = np.array([[np.cos(ori1), np.sin(ori1)]]) @ self.center / sf * 2 * np.pi
        _dist_ϕ_2 = np.array([[np.cos(ori2), np.sin(ori2)]]) @ self.center / sf * 2 * np.pi + Δphase
        
        _vec = contrast1 * np.exp(_dist_ϕ_1[0][0] * 1j) + contrast2 * np.exp(_dist_ϕ_2[0][0] * 1j)
        _luminance = np.sin(np.angle(_vec) + phase)

        return _luminance * 2, np.abs(_vec)
    
    def get_response_grating(self, *args, **kwargs):
        """calculate the linear-nonlinear response of subunit to sinusoidal grating stimulus.
        
        #TODO: formula to calculate the phase and contrast level.
        
        Keyword Arguments:
        - ori: orientation in rad [default: 0]
        - contrast: contrast level in [0, 1] [default: 0.48]
        - phase: `real` or `numpy.ndarray` phase in rad [default: 0]
        - sf: spatial frequency in arbitrary unit per cycle [default: 50]
        """
        _l, _c = self._get_sinusoidal(*args, **kwargs)
        return self.nonlinear(self._get_contrast_response(_c) * _l)
    
    def get_response_plaid(self, *args, **kwargs):
        """calculate the linear-nonlinear response of subunit to sinusoidal plaid stimulus.
        
        #TODO: formula to calculate the phase and contrast level.
        
        Keyword Arguments:
        - ori1: primary orientation in rad [default: 0]
        - ori2: secondary orientation in rad [default: 0]
        - contrast1: primary contrast level in [0, 1] [default: 0.48]
        - contrast2: secondary contrast level in [0, 1] [default: 0.48]
        - phase: `real` or `numpy.ndarray` phase in rad [default: 0]
        - Δphase: phase offset of secondary grating [default: 0]
        - sf: spatial frequency in arbitrary unit per cycle [default: 50]
        """
        _l, _c = self._get_sinusoidal_plaid(*args, **kwargs)
        return self.nonlinear(self._get_contrast_response(_c) * _l)


class V1node(object):
    """LN model for V1 neuron
    
    This class will group a list of LGNnode and process the convergent inputs from LGN subunits.
    
    Parameters:
    - `subunits`: list of `LGNunit`s [default: []]
    - `nonlinear`: the nonlinear function [default: relu]
    
    Public Methods:
    - `get_response_grating`
    - `get_response_plaid`
    """
    def __init__(self, subunits=[], nl=relu):
        """Constructor of `V1node` class
        
        Keyword Arguments:
        - subunits: list of `LGNnode`s [default: []]
        - nl: `function` nonlinearity function [default: relu]
        """
        super(V1node, self).__init__()
        self.subunits = subunits
        self.nonlinear = nl
            
    def get_response_grating(self, length=720, phase=0, **kwargs):
        """LN resposne to the convergent inputs from LGN subunits to sinusoidal grating stimulus.
        
        Return the response of one full cycle of grating stimulus given parameters.
        
        Keyword Arguments:
        - length: `int` total steps for simulating the full cycle [default: 720]
        - phase: `real` the initial phase value [default: 0]
        - **kwargs: refer to `LGNnode.get_response_grating` for full documentation.
        """
        _resp = np.zeros(length)
        _step = np.linspace(0, 2*np.pi*(1-1/length), length)
        for item in self.subunits:
            _resp += item.get_response_grating(phase=_step+phase, **kwargs)
        return _resp / len(self.subunits)
    
    def get_response_plaid(self, length=720, phase=0, **kwargs):
        """LN resposne to the convergent inputs from LGN subunits to sinusoidal plaid stimulus.
        
        Return the response of one full cycle of plaid stimulus given parameters.
        
        Keyword Arguments:
        - length: `int` total steps for simulating the full cycle [default: 720]
        - phase: `real` the initial phase value [default: 0]
        - **kwargs: refer to `LGNnode.get_response_grating` for full documentation.
        """
        _resp = np.zeros(length)
        _step = np.linspace(0, 2*np.pi*(1-1/length), length)
        for item in self.subunits:
            _resp += item.get_response_plaid(phase=_step+phase, **kwargs)
        return _resp / len(self.subunits)
