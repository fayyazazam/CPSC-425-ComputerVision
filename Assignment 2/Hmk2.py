from PIL import Image
import numpy as np
import math
from scipy import signal

def boxfilter( n ):
    #Check if its even, throw an error if it is
    #Box filter should be odd, to have better centre precision
    if (n % 2 ==0):
        return AssertionError('Dimension must be odd')
    else:
        #All values in array should sum up to equal 1
        #Thats why we divide by 1/n^2 to get the number
        arrayNumber = 1/math.pow(n, 2)
        return np.full((n, n), arrayNumber)

def gauss1d(sigma):
    #As per instructions, length should be 6 times sigma and rounded up
    arrayLength = math.ceil(sigma*6)
    #in case it's an even number add one to make it odd
    if arrayLength % 2 == 0:
        arrayLength = arrayLength+1
        
    #Want to generate a 1D array, centred at 0, but going negative
    #and positive in each relative direction. 

    #For example: sigma 1.0 will have sigma*6: [-3, -2, -1, 0, 1, 2, 3]
    #Hence it goes up to half the array length (3) for each side of 0
    gaussArray = np.arange(-(math.floor(arrayLength/2)), 
        math.ceil(arrayLength/2))
        
    #For each number in array, we want to apply the gaussian function
    #dependant on their length from the middle
    result = map(lambda x: round(math.exp(-((x**2)/(2*(sigma**2)))), 8), gaussArray)
    #Normalize each value with the constant of 1/(sum of the array)
    normalizedResult = np.array(result)*(1/(np.sum(result)))
    return normalizedResult
    
def gauss2d(sigma):
    #Find the 1D gaussian given sigma
    gauss1Array = gauss1d(sigma)
    #Make it 2D by adding an axis, so we can transpose it
    gauss1Array = gauss1Array[np.newaxis]
    #Transpose the 1D (now 2D) array
    gauss1Transpose = np.transpose(gauss1Array)
    
    #Convolve the 2D array with its transpose
    return signal.convolve2d(gauss1Array, gauss1Transpose)
    
def gaussconvolve2d(array, sigma):
    #Get the convolution given sigma
    ifilter = gauss2d(sigma)
    #Apply the filter (convolution) to image (array)
    return signal.convolve2d(array, ifilter, 'same')
    
    
    