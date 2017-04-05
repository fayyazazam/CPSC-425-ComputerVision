from PIL import Image, ImageDraw
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Computes the cost of given boundaries. Good boundaries have zero cost.
def get_boundaries_cost( boundaries, good_boundaries ):
	return np.sum( boundaries != good_boundaries );

# Finds the indices of color_histograms given a series of cluster centres.
def cluster2boundaries(histograms, centres):

	# Find the cluster assignment of each histogram
	distances = cdist( histograms, centres )
	idx       = np.argmin( distances, 1 )

	# Find the points where the index changes
	boundaries = np.zeros( len(idx)+1, dtype = np.bool )

	for i in range( len(idx)-1 ):
		boundaries[i+1] = idx[i] != idx[i+1];

	return boundaries

# Computes histograms from gray images
def compute_gray_histograms( grays, nbins ):
	gray_hs = np.zeros(( nframes, nbins ), dtype = np.uint16 );

	for i in range( len(grays) ):
		gray_im = grays[i]
		v1 = np.histogram(gray_im.flatten(),bins=nbins, range=(0,255))
		gray_hs[i] = v1[0]

	return gray_hs;


def compute_color_histograms( colors, nbins ):
	# === WRITE THE FUNCTION HERE ===

	#Initialize R,B,G histograms
	red_histogram = np.zeros(( nframes, nbins), dtype=np.uint16)
	green_histogram = np.zeros(( nframes, nbins), dtype=np.uint16)
	blue_histogram = np.zeros(( nframes, nbins), dtype=np.uint16)

	for i in range(len(colors)):
		# Slice by the 3 color channels

		# Slice by red
		red_color = colors[i,:,:,0] 
		# Slice by green
		green_color = colors[i,:,:,1]
		# Slice by blue
		blue_color = colors[i,:,:,2]

		#Calculate histograms on the sliced images (flattened)
		red = np.histogram(red_color.flatten(), bins=nbins, range=(0, 255))
		green = np.histogram(red_color.flatten(), bins=nbins, range=(0, 255))
		blue = np.histogram(red_color.flatten(), bins=nbins, range=(0, 255))

		#Store the calculated values
		red_histogram[i] = red[0]
		green_histogram[i] = green[0]
		blue_histogram[i] = blue[0]

	# Combine them into one
	color_histogram = np.hstack((red_histogram, green_histogram, blue_histogram))
	return color_histogram


# === Main code starts here ===
fname     = 'colours' # folder name 
nframes   = 151       # number of frames
im_height = 90        # image height 
im_width  = 120       # image width

# define the list of (manually determined) shot boundaries here
good_boundaries = [33, 92, 143];

# convert good_boundaries list to a binary array
gb_bool = np.zeros( nframes+1, dtype = np.bool )
gb_bool[ good_boundaries ] = True

# Create some space to load the images into memory
colors = np.zeros(( nframes, im_height, im_width, 3), dtype = np.uint8);
grays  = np.zeros(( nframes, im_height, im_width   ), dtype = np.uint8);

# Read the images and store them in color and grayscale formats
for i in range( nframes ):
	imname    = '%s/dwc%03d.png' % ( fname, i+1 )
	im        = Image.open( imname ).convert( 'RGB' )
	colors[i] = np.asarray(im, dtype = np.uint8)
	grays[i]  = np.asarray(im.convert( 'L' ))

# Initialize color histogram
nclusters   = 4;
nbins       = range(2,13)
gray_costs  = np.zeros( len(nbins) );
color_costs = np.zeros( len(nbins) );

# === GRAY HISTOGRAMS ===
for n in nbins:
	# Compute the gray histogram first
	gray_histogram = compute_gray_histograms(grays, n)
	gray_histogram = gray_histogram.astype(float)
	# Compute K-means
	k_means = kmeans(gray_histogram, nclusters)
	# Compute boundaries
	bounds = cluster2boundaries(gray_histogram, k_means[0])
	print bounds
	print k_means
	# Get cost
	cost = get_boundaries_cost(bounds, gb_bool)
	# Store cost
	gray_costs[n-2] = cost
# === END GRAY HISTOGRAM CODE ===

plt.figure(1);
plt.xlabel('Number of bins')
plt.ylabel('Error in boundary detection')
plt.title('Boundary detection using gray histograms')
plt.plot(nbins, gray_costs)
plt.axis([2, 13, -1, 10])
plt.grid(True)
plt.show()

# === COLOR HISTOGRAMS ===
for n in nbins:
	# Compute the color histogram first
	color_histogram = compute_color_histograms(colors, n)
	color_histogram = color_histogram.astype(float)
	# Compute K-means
	k_means = kmeans(color_histogram, nclusters)
	# Compute boundaries
	color_bounds = cluster2boundaries(color_histogram, k_means[0])
	# Get cost
	cost = get_boundaries_cost(color_bounds, gb_bool)
	# Store cost
	color_costs[n-2] = cost
# === END COLOR HISTOGRAM CODE ===

plt.figure(2);
plt.xlabel('Number of bins')
plt.ylabel('Error in boundary detection')
plt.title('Boundary detection using color histograms')
plt.plot(nbins, color_costs)
plt.axis([2, 13, -1, 10])
plt.grid(True)
plt.show()

fdiffs = np.zeros( nframes )
# === ABSOLUTE FRAME DIFFERENCES ===
temp = 0
for n in range(nframes):
	# Skip first as it doesn't have anything to compare to
	if n>0:
		# Get the previous value
		temp = grays[n-1]
		# Get current value
		current = grays[n]
		# Calculate difference
		difference = np.sum(abs(current - temp))
		# Store difference
		fdiffs[n] = difference
	pass
plt.figure(4)
plt.xlabel('Frame number')
plt.ylabel('Absolute frame difference')
plt.title('Absolute frame differences')
plt.plot(fdiffs)
plt.show()

sqdiffs = np.zeros( nframes )
# === SQUARED FRAME DIFFERENCES ===
temp = 0
for n in range(nframes):
	# Skip first as it doesn't have anything to compare to
	if n>0:
		# Get the previous value
		temp = grays[n-1]
		# Get current value
		current = grays[n]
		# Calculate difference
		difference = np.sum(pow((current - temp), 2))
		# Store difference
		sqdiffs[n] = difference
	pass

plt.figure(5)
plt.xlabel('Frame number')
plt.ylabel('Squared frame difference')
plt.title('Squared frame differences')
plt.plot(sqdiffs)
plt.show()

avgdiffs = np.zeros( nframes )
# === AVERAGE GRAY DIFFERENCES ===
for n in range(nframes):
	# Skip first as it doesn't have anything to compare to
	if n>0:
		# Get the previous value
		temp = grays[n-1]   
		# Get current value
		current = grays[n]     
		# Calculate the difference in the average
		difference = np.average(current) - np.average(temp)
		# Store the difference
		avgdiffs[n] = difference
	pass

plt.figure(6)
plt.xlabel('Frame number')
plt.ylabel('Average gray frame difference')
plt.title('Average gray frame differences')
plt.plot(avgdiffs)
plt.show()

histdiffs = np.zeros( nframes )
# === HISTOGRAM DIFFERENCES ===
for n in range(nframes):
	# Skip first as it doesn't have anything to compare to
	if n>0:
		# Get the previous value
		temp = grays[n-1]
		# Get current value
		current = grays[n]
		# Gray histogram of previous value with 10 bins
		previous_histogram = compute_gray_histograms(temp, 10)
		# Gray histogram of currentvalue with 10 bins  
		current_histogram = compute_gray_histograms(current, 10)
		# Calculate distance
		distance = np.linalg.norm(current_histogram - previous_histogram)
		# Store the value
		histdiffs[n] = distance 
	pass

plt.figure(7)
plt.xlabel('Frame number')
plt.ylabel('Histogram frame difference')
plt.title('Histogram frame differences')
plt.plot(histdiffs)
plt.show()
