import hmk3
from PIL import Image

#Load an image from the ones given
im = Image.open("faces/family.jpg")

#Build pyramid with minsize = 20
pyramid = hmk3.MakePyramid(im, 20)

#Show and save pyramid image
hmk3.ShowPyramid(pyramid)