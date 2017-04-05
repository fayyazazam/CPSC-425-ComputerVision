import hmk3
from PIL import Image


#THRESHOLD:
threshold = 0.52

# loads the template
template = Image.open("faces/template.jpg")
# resize factor
RF = int(template.size[1]*15/template.size[0])
# resizes the template
template = template.resize((15, RF), Image.BICUBIC)
# loads the image where the match will be performed
image = Image.open("faces/tree.jpg")
# builds the pyramid
pyramid = hmk3.MakePyramid(image, 15)
# finds the correlation between pyramid and template
thresholded_match_list = hmk3.FindTemplate(pyramid, template, threshold)
# draw matches on the original image
hmk3.DrawMatch(pyramid, template, thresholded_match_list)