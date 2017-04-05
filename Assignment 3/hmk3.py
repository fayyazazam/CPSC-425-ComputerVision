from PIL import Image, ImageDraw
import numpy as np
import ncc

#Q2
def MakePyramid(image, minsize):
	#Create array of images for pyramid
	#With first image loaded in already
	pyramid = [image]

	#Get the x value of image size to compare with minsize
	size = image.size[0]

	while size >= minsize:
		#Get the last pyramid that was added to the list
		#Which we will use to resize by rfactor
		lastImage = pyramid[-1]

		#Get the x, y of this last image
		(x, y) = lastImage.size

		#Append the resized image to the pyramid array
		pyramid.append(lastImage.resize((int(x*0.75), int(y*0.75)),
			Image.BICUBIC))

		#Get the last x value of image so our loop can
		#Get killed or continue
		size = int(lastImage.size[0])

	return pyramid

#Q3
def ShowPyramid(pyramid):
	#Get the full width we need the image to be initalized to
	#By adding up all the widths in the pyramid array
	pyramidWidth = sum([img.size[0] for img in pyramid][:-1])

	#Create the canvas on which we will paste images
	#From pyramid
	canvas = Image.new("L", (pyramidWidth, pyramid[0].size[1]), "white")

	#Need an offset variable to place images side by side
	offset_x = 0

	for img in pyramid:
		#Paste images side by side on canvas
		canvas.paste(img, (offset_x, pyramid[0].size[1]-img.size[1]))
		#Increase image offset for next image
		offset_x += img.size[0]
	canvas.show()
	canvas.save('pyramid.png', 'PNG')

#Q4a
def FindTemplate(pyramid, template, threshold):
	matches = []

	for img in pyramid:
		matches.append(ncc.normxcorr2D(img, template))

	thresholdMatches = []

	#For the matched images in pyramid, diregard the ones
	#With small correlations
	for match in matches:
		thresholdMatches.append(np.where(match >= threshold, 1, 0))

	return thresholdMatches

#Q4b
def DrawMatch(pyramid, template, image_arrays):
	#Convert image to color, so we can put the red rectangles
	im = pyramid[0].convert("RGB")
	draw = ImageDraw.Draw(im)

	#Current Image
	currImgIndex = 0
	
	#Points with correlation > threshold
	pointsList = []
	
	#Size of template
	(i, j) = template.size

	for image in image_arrays:
		#Get coords for high correlation points
		pointsList = np.nonzero(image)
		#Resize red boxes for dimensions dependant on which image
		i /= 0.75 ** currImgIndex
		j /= 0.75 ** currImgIndex

		#Draw rectangle centred at correlation point
		for p in range(len(pointsList[0])):
			#Resize point coordinates based on size of current img
			x = pointsList[1][p] / (0.75) ** currImgIndex
			y = pointsList[0][p] / (0.75) ** currImgIndex

			x1 = x-i/2
			y1 = y-j/2
			x2 = x+i/2
			y2 = y+j/2

			#Draw 4 lines to make rectangle
			draw.line((x1, y1, x1, y2), fill="red", width=2)
			draw.line((x1, y1, x2, y1), fill="red", width=2)
			draw.line((x1, y2, x2, y2), fill="red", width=2)
			draw.line((x2, y1, x2, y2), fill="red", width=2)

		currImgIndex += 1
	del draw
	im.show()
	im.save('PicMatch1.png', 'PNG')










