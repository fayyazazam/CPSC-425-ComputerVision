import Hmk2
from PIL import Image
import numpy as np

print("1) BOXFILTER TESTS: \n")
print(Hmk2.boxfilter(3))
"""
	[[ 0.11111111  0.11111111  0.11111111]
	 [ 0.11111111  0.11111111  0.11111111]
	 [ 0.11111111  0.11111111  0.11111111]]
"""
print(Hmk2.boxfilter(4))
"""
	AssertionError: Dimension must be odd
"""
print(Hmk2.boxfilter(5))
"""
	[[ 0.04  0.04  0.04  0.04  0.04]
	 [ 0.04  0.04  0.04  0.04  0.04]
	 [ 0.04  0.04  0.04  0.04  0.04]
	 [ 0.04  0.04  0.04  0.04  0.04]
	 [ 0.04  0.04  0.04  0.04  0.04]]
"""
	


print("\n\n\n2) GAUSS1D TESTS: \n")
print(Hmk2.gauss1d(0.3))
"""
	[ 0.00383626  0.99232748  0.00383626]
"""
print(Hmk2.gauss1d(0.5))
"""
	[ 0.10650698  0.78698605  0.10650698]
"""
print(Hmk2.gauss1d(1))
"""
	[ 0.00443305  0.05400558  0.24203623  0.39905028  0.24203623  0.05400558
	  0.00443305]
"""
print(Hmk2.gauss1d(2))
"""
	[ 0.0022182   0.00877313  0.02702316  0.06482519  0.12110939  0.17621312
	  0.19967563  0.17621312  0.12110939  0.06482519  0.02702316  0.00877313
	  0.0022182 ]
"""



print("\n\n\n3) GAUSS2D TESTS: \n")
print(Hmk2.gauss2d(0.5))
"""
	[[ 0.01134374  0.0838195   0.01134374]
	 [ 0.0838195   0.61934704  0.0838195 ]
	 [ 0.01134374  0.0838195   0.01134374]]
"""
print(Hmk2.gauss2d(1.0))
"""
	[[  1.96519284e-05   2.39409418e-04   1.07295860e-03   1.76900966e-03
	    1.07295860e-03   2.39409418e-04   1.96519284e-05]
	 [  2.39409418e-04   2.91660281e-03   1.30713073e-02   2.15509423e-02
	    1.30713073e-02   2.91660281e-03   2.39409418e-04]
	 [  1.07295860e-03   1.30713073e-02   5.85815363e-02   9.65846250e-02
	    5.85815363e-02   1.30713073e-02   1.07295860e-03]
	 [  1.76900966e-03   2.15509423e-02   9.65846250e-02   1.59241126e-01
	    9.65846250e-02   2.15509423e-02   1.76900966e-03]
	 [  1.07295860e-03   1.30713073e-02   5.85815363e-02   9.65846250e-02
	    5.85815363e-02   1.30713073e-02   1.07295860e-03]
	 [  2.39409418e-04   2.91660281e-03   1.30713073e-02   2.15509423e-02
	    1.30713073e-02   2.91660281e-03   2.39409418e-04]
	 [  1.96519284e-05   2.39409418e-04   1.07295860e-03   1.76900966e-03
	    1.07295860e-03   2.39409418e-04   1.96519284e-05]]
"""


print("\n\n\n4) GAUSS2D TESTS: \n")
im = Image.open('van.png')
#Open original image
im.show()

#Convert to grayscale
im = im.convert('L')
im_array = np.asarray(im)

#Apply the filter using the gaussconvolve2d function we coded
imconvolved = Hmk2.gaussconvolve2d(im_array, 3)
#Convert the type or PIL throws as an error
imconvolvedImage = Image.fromarray(imconvolved.astype(np.uint8))
#Save the filtered image
imconvolvedImage.save('van-filtered.png', 'PNG')

#Open filtered image
imconvolvedImage.show()