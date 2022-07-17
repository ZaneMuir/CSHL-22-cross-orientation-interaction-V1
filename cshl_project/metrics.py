import numpy as np

class IndexCalc(object):
    """Static class for various indices.
    
    Available Methods:
    - `masking_index`
    - `selectivity_index`
    - `F1_modulation`
    """
    @staticmethod
    def masking_index(test, mask, plaid):
        """Masking index defined as:
        
        .. math::
            \\text{MI} = \\frac{\\text{Plaid} - (\\text{Test} + \\text{Mask})}{\\text{Plaid} + (\\text{Test} + \\text{Mask})}
        """
        _r = test + mask
        return (plaid - _r) / (plaid + _r)
    
    @staticmethod
    def selectivity_index(test, mask):
        """Selectivity index defined as:
        
        .. math::
            \\text{SI} = \\frac{\\text{Test} - \\text{Mask}}{\\text{Test} + \\text{Mask}}
        """
        return (test - mask) / (test + mask)
    
    @staticmethod
    def F1_modulation(response, reference=None, order=1):
        """F1 modulation component using Fourier transform."""
        if reference is None:
            return np.abs(np.fft.fft(response))[order] / len(response) * 2
        else:
            return IndexCalc.F1_modulation(response, order=order) / IndexCalc.F1_modulation(reference, order=order)
