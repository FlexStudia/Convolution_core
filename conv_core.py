# coding: utf-8

"""
This code is a module for calculating the convolution of spectra.

An example of its use can be found at the end — the demo function.

How to use the module:
0. Import the module into your code and satisfy the dependencies (math, numpy, scipy).
    from conv_core import ConvolutionCore

1. Provide input data that satisfies the conditions of applicability.
    1. High-resolution spectrum
        - spectrum_wavelength
            Contains high-resolution spectrum wavelengths.
            Must be a NumPy array.
            Its length must be equal to that of spectrum_intensity.
        - spectrum_intensity
            Contains high-resolution spectrum intensities.
            Must be a NumPy array.
            Its length must be equal to that of spectrum_wavelength.
    2. Target - or destination - wavelength
        - destination_wavelength
            Contains low-resolution - or target, or destination - spectrum wavelengths.
            Must be a NumPy array.
            Its values must not exceed those of the spectrum_wavelength.
    3. Convolution function parameters
        - convolution_type
            Determines the type of convolution function.
            Must be an integer between 0 and 5, inclusive.
            0 is for the Gauss function;
            1,       the triangle function;
            2,       the trapeze function;
            3,       the Lorentz function;
            4,       the Voigt function; and
            5,       the sum of the Gauss and Lorentz functions.
        - convolution_truncation
            Determines when to truncate the sum during convolution calculation.
            Must be a positive float.
            The convolution_truncation and every width in convolution_width_1 and convolution_width_2 must satisfy the condition:
            convolution_truncation > n * width, where n is
            3    for the Gauss function,
            1    for the triangle and trapeze functions,
            2000 for the Lorentz function, the Voigt function and sum of the Gauss and Lorentz functions.
        - convolution_gauss_ratio
            Determines the ratio between the Gauss and Lorentz functions in their sum: r * G + (1-r) * L,
            where r is convolution_gauss_ratio, G is the Gauss function, and L is the Lorentz function.
            Used only for convolution_type 5 (for the sum of the Gauss and Lorentz functions).
            Must be a positive float between 0 and 1, inclusive.
            For all other convolution functions, it is not applicable and can be set to 0.
        - convolution_width_1
            Determines the widths of the destination spectrum.
            Must be a NumPy array.
            Its length must be equal to that of destination_wavelength.
            Each width must exceed the corresponding step in the destination_wavelength.
        - convolution_width_2
            Determines the widths of the destination spectrum.
            Only used for convolution_type 2 (the trapeze function).
            It is known as the top of the trapeze function.
            Must be a NumPy array or numpy.nan.
            If used (the trapeze function), its length must equal the lengths of destination_wavelength and convolution_width_1.
            If used (the trapeze function), each width must exceed the corresponding step in the destination_wavelength.

2. Create an instance of the ConvolutionCore class and run the calculation.
    my_conv = ConvolutionCore(spectrum_wavelength, spectrum_intensity,
                              destination_wavelength,
                              convolution_type, convolution_truncation, convolution_gauss_ratio,
                              convolution_width_1, convolution_width_2)
    my_conv.convolution_calc()

3. Call the getter to obtain the calculation result.
    my_conv.class_getter()
        It returns the low-resolution - or target, or destination - spectrum intensities.
"""

# IMPORTS
import numpy as np
import math
from scipy import special


# STATIC functions
def is_array_empty(arr):
    if not isinstance(arr, np.ndarray) and not isinstance(arr, list):
        return True
    if type(arr) == list:
        if arr == [] or arr == [0] * len(arr):
            return True
    if isinstance(arr, np.ndarray):
        if np.array_equal(arr, np.array([])) or np.array_equal(arr, np.zeros(len(arr))):
            return True
    return False

def gauss_f(mean, sigma, x):
    try:
        if sigma == 0:
            sigma = 10 ** -15
        return math.exp(- ((x - mean) ** 2) / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    except Exception as e:
        print(f"Error in gauss_f: {e}")
        return np.nan

def lorentz_f(mean, sigma, x):
    try:
        if sigma == 0:
            sigma = 10 ** -15
        return 1 / (math.pi * sigma * (1 + ((x - mean) ** 2 / (sigma ** 2))))
    except Exception as e:
        print(f"Error in lorentz_f: {e}")
        return np.nan

def gauss_lorentz_f(mean, sigma_gauss, sigma_lorentz, x, gauss_ratio):
    try:
        return gauss_ratio * gauss_f(mean, sigma_gauss, x) + (1 - gauss_ratio) * lorentz_f(mean, sigma_lorentz, x)
    except Exception as e:
        print(f"Error in gauss_lorentz_f: {e}")
        return np.nan

def voigt_f(mean, sigma_gauss, sigma_lorentz, x):
    try:
        if sigma_gauss == 0:
            sigma_gauss = 10 ** -15
        z = (x - mean + sigma_lorentz * 1j) / (sigma_gauss * math.sqrt(2))
        return special.wofz(z).real / (sigma_gauss * math.sqrt(2 * math.pi))
    except Exception as e:
        print(f"Error in voigt_f: {e}")
        return np.nan

def triangle_f(mean, sigma, x):
    try:
        if sigma == 0:
            sigma = 10 ** -15
        if np.fabs(x - mean) > 2 * sigma:
            return 10 ** -15
        return (- np.fabs(mean - x) + 2 * sigma) / (4 * sigma ** 2)
    except Exception as e:
        print(f"Error in triangle_f: {e}")
        return np.nan

def trapeze_f(mean, sigma, top, x):
    try:
        if sigma == 0:
            sigma = 10 ** -15
        if math.fabs(x - mean) > 2 * sigma - top:
            return 10 ** -15
        if math.fabs(x - mean) <= top:
            return 1 / (2 * sigma)
        return (- np.fabs(mean - x) + 2 * sigma - top) / (4 * sigma * (sigma - top))
    except Exception as e:
        print(f"Error in trapeze_f: {e}")
        return np.nan


# MAIN CLASS
class ConvolutionCore:
    def __init__(self, spectrum_wavelength, spectrum_intensity,
                 destination_wavelength,
                 convolution_type, convolution_truncation, convolution_gauss_ratio,
                 convolution_width_1, convolution_width_2):
        self.init_globals()
        self.class_setter(spectrum_wavelength, spectrum_intensity,
                           destination_wavelength,
                           convolution_type, convolution_truncation, convolution_gauss_ratio,
                          convolution_width_1, convolution_width_2)

    def init_globals(self):
        # spectrum
        self.spectrum_wavelength = np.zeros(0)
        self.spectrum_intensity = np.zeros(0)
        # destination
        self.destination_wavelength = np.zeros(0)
        self.destination_intensity = np.zeros(0)
        # convolution function
        self.convolution_type = -1 # 0: Gauss, 1: triangle, 2: trapeze, 3: Lorentz, 4: Voigt, 5: Gauss+Lorentz
        self.convolution_truncation = 0
        self.convolution_gauss_ratio = 0
        self.convolution_width_1 = np.zeros(0)
        self.convolution_width_2 = np.zeros(0)

    def class_setter(self, spectrum_wavelength, spectrum_intensity,
                     destination_wavelength,
                     convolution_type, convolution_truncation, convolution_gauss_ratio,
                     convolution_width_1, convolution_width_2):
        self.spectrum_wavelength = spectrum_wavelength
        self.spectrum_intensity = spectrum_intensity
        self.destination_wavelength = destination_wavelength
        self.destination_intensity = np.zeros(len(self.destination_wavelength))
        self.convolution_type = convolution_type
        self.convolution_truncation = convolution_truncation
        self.convolution_gauss_ratio = convolution_gauss_ratio
        self.convolution_width_1 = convolution_width_1
        self.convolution_width_2 = convolution_width_2

    def convolution_calc(self):
        for w in range(len(self.destination_wavelength)):
            # get sigma_1 & 2
            sigma_1, sigma_2 = self.get_sigmas(w)
            # get j_min & j_max in self.spectrum_wavelength
            j_min, j_max = self.get_wavelength_boundaries(sigma_1, sigma_2, w)
            # apply convolution function
            truncated_convolution_function = self.convolution_profil_calc(w, j_min, j_max, sigma_1, sigma_2)
            # normalization
            normalized_convolution_function = self.normalization(truncated_convolution_function, j_min, j_max)
            del truncated_convolution_function
            # convolution
            self.convolution_finalization(w, normalized_convolution_function, j_min, j_max)

    def get_sigmas(self, w):
        sigma_factors = {0: (2.3548, 1),  # "Gauss"
                         1: (2, 1),  # "triangle"
                         2: (2, 2),  # "trapeze"
                         3: (2, 1),  # "Lorentz"
                         4: (2.3548, 2),  # "Voigt"
                         5: (2.3548, 2),  # "Gauss+Lorentz"
                         }
        sigma_1 = self.convolution_width_1[w] / sigma_factors[self.convolution_type][0]
        sigma_2 = 0 if is_array_empty(self.convolution_width_2) else self.convolution_width_2[w] / sigma_factors[self.convolution_type][1]
        return sigma_1, sigma_2

    def get_wavelength_boundaries(self, sigma_1, sigma_2, w):
        j_min = 0
        j_max = len(self.spectrum_wavelength) - 1
        if self.convolution_type not in [2, 4, 5]:  # Gauss function, triangle, Lorentz function
            for j in range(len(self.spectrum_wavelength)):
                if self.spectrum_wavelength[j] > self.destination_wavelength[w] - self.convolution_truncation * sigma_1 and j_min == 0:
                    j_min = j
                if self.spectrum_wavelength[j] > self.destination_wavelength[w] + self.convolution_truncation * sigma_1 and j_max == len(self.spectrum_wavelength) - 1:
                    j_max = j
                    break
        elif self.convolution_type == 2:  # trapeze
            for j in range(len(self.spectrum_wavelength)):
                if self.spectrum_wavelength[j] > self.destination_wavelength[w] - sigma_2 - self.convolution_truncation * sigma_1 and j_min == 0:
                    j_min = j
                if self.spectrum_wavelength[j] > self.destination_wavelength[w] + sigma_2 + self.convolution_truncation * sigma_1 and j_max == len(self.spectrum_wavelength) - 1:
                    j_max = j
                    break
        elif self.convolution_type in [4, 5]:  # Voigt, Gauss+Lorentz
            for j in range(len(self.spectrum_wavelength)):
                if self.spectrum_wavelength[j] > self.destination_wavelength[w] - self.convolution_truncation * (sigma_1 + sigma_2) and j_min == 0:
                    j_min = j
                if self.spectrum_wavelength[j] > self.destination_wavelength[w] + self.convolution_truncation * (sigma_1 + sigma_2) and j_max == len(self.spectrum_wavelength) - 1:
                    j_max = j
                    break
        return j_min, j_max

    def convolution_profil_calc(self, w, j_min, j_max, sigma_1, sigma_2):
        index = 0
        truncated_convolution_function = np.zeros(j_max - j_min)
        methods = {
            0: gauss_f,
            1: triangle_f,
            2: trapeze_f,
            3: lorentz_f,
            4: voigt_f,
            5: gauss_lorentz_f
        }
        for j in range(j_min, j_max):
            if self.convolution_type == 2 or self.convolution_type == 4:
                truncated_convolution_function[index] = methods[self.convolution_type](self.destination_wavelength[w], sigma_1, sigma_2, self.spectrum_wavelength[j])
            elif self.convolution_type == 5:
                truncated_convolution_function[index] = methods[self.convolution_type](self.destination_wavelength[w], sigma_1, sigma_2, self.spectrum_wavelength[j], self.convolution_gauss_ratio)
            else:
                truncated_convolution_function[index] = methods[self.convolution_type](self.destination_wavelength[w], sigma_1, self.spectrum_wavelength[j])
            index += 1
        return truncated_convolution_function

    def normalization(self, truncated_conv_func, j_min, j_max):
        area_under_curve = np.trapezoid(truncated_conv_func, self.spectrum_wavelength[j_min:j_max])
        return truncated_conv_func / area_under_curve

    def convolution_finalization(self, w, normalized_convolution_function, j_min, j_max):
        if sum(normalized_convolution_function):
            self.destination_intensity[w] = np.convolve(self.spectrum_intensity[j_min:j_max], np.flip(normalized_convolution_function), "valid")[0] / sum(normalized_convolution_function)
        else:
            closest_index = 0
            delta_value = math.fabs(self.destination_wavelength[w] - self.spectrum_wavelength[0])
            for j in range(len(self.spectrum_wavelength)):
                if delta_value < math.fabs(self.destination_wavelength[w] - self.spectrum_wavelength[j]):
                    closest_index = j
                    delta_value = math.fabs(self.destination_wavelength[w] - self.spectrum_wavelength[j])
            self.destination_intensity[w] = self.spectrum_intensity[closest_index]

    def class_getter(self):
        return self.destination_intensity


def demo():
    # inputs
    spectrum_wavelength = np.arange(-12.41, 25.26, 0.01)
    spectrum_intensity = spectrum_wavelength * spectrum_wavelength
    destination_wavelength = np.arange(0, 10.5, 0.5)
    convolution_type = 0
    convolution_truncation = 10
    convolution_gauss_ratio = 0
    convolution_width_1 = np.full(len(destination_wavelength), 1)
    convolution_width_2 = np.nan
    # class evocation
    my_conv = ConvolutionCore(spectrum_wavelength, spectrum_intensity,
                              destination_wavelength,
                              convolution_type, convolution_truncation, convolution_gauss_ratio,
                              convolution_width_1, convolution_width_2)
    my_conv.convolution_calc()
    # outputs
    print(my_conv.class_getter())

# uncomment to run the example
#demo()