import tifffile as tf
import napari as nap
import numpy as np
import matplotlib.pylab as plt
import csv, statistics
from csv import writer
from math import sqrt
from skimage.feature import blob_log

def double_thresholding(image, lower_threshold, upper_threshold):
    """
    Apply double thresholding to an image.
    
    Parameters:
    - image: The input image array.
    - lower_threshold: The lower threshold value.
    - upper_threshold: The upper threshold value.
    
    Returns:
    - binary_image: The resulting binary image after thresholding.
    """
    binary_image = np.logical_and(image > lower_threshold, image < upper_threshold)
    return binary_image

def find_centre(coords1, coords2, coords3, filename):
    """
    Find the center of a circle given three points on its circumference.

    Parameters:
    - coords1, coords2, coords3: Tuples representing the (x, y) coordinates of the points.

    Returns:
    - cx, cy: The (x, y) coordinates of the circle's center.
    - stores midpoints values and gradient/y-intercept values.
    """
    x1, y1 = coords1
    x2, y2 = coords2
    x3, y3 = coords3

    # Find the line bisector between coords1 and coords2
    m1 = -((x1 - x2) / (y1 - y2))
    mx1, my1 = (x1 + x2) / 2, (y1 + y2) / 2
    c1 = my1 - (m1 * mx1)

    # Find the line bisector between coords2 and coords3
    m2 = -((x2 - x3) / (y2 - y3))
    mx2, my2 = (x2 + x3) / 2, (y2 + y3) / 2
    c2 = my2 - (m2 * mx2)

    # Find the center
    cx = (c2 - c1) / (m1 - m2)
    cy = (m1 * cx) + c1
    
    #saving the values so i make linear equation to show in napari
    with open(filename, mode='a', newline='') as f:
        data=[mx1,my1,mx2,my2,m1,c1,m2,c2]
        writer_obj = writer(f)
        writer_obj.writerow(data)
    
    return cx, cy 

def frame(step, end):
    while end < 2200:
        end += step
    return (end-(step-1))



filename = 'BisectorEqua.csv'
with open(filename, mode='w', newline='') as f:     #creates a csv file and adds the fieldnames
    fields = ['mx1','my1','mx2','my2','m1','c1','m2','c2']
    writer_obj = csv.DictWriter(f,fieldnames=fields)
    writer_obj.writeheader()

#opens the tiff file
rotation_increment_data = tf.imread('example_derotation_data/imaging/rotation_increment_00001.tif')

#thresholds
lower_threshold = -3325
upper_threshold = -2150

#frames
start = 70
step = 95
last_frame = frame(step,start)

centre = [] # stores coordinates of the centre
coords = [] # coordinates of the blobs

midpoint = [] # stores the midpoints between blobs: 'mx1','my2','mx2','my2'
linear = [] # stores the gradient and the y-intercept of 'm1','c1','m2','c2'

indices = list(range(start, last_frame, step))
subset_rotation_increment_data = rotation_increment_data[indices]

binary_rotation_increment_data = np.array([double_thresholding(img, lower_threshold, upper_threshold) for img in subset_rotation_increment_data])

blobs = [blob_log(img, max_sigma=40, min_sigma=10, threshold=0.11) for img in binary_rotation_increment_data]

for b in blobs:
    if len(b) != 0:
        if not(any(np.array_equal(b, coord) for coord in coords)) and (30<sqrt((128.-(b[0][0]))**2+(128.-(b[0][1]))**2)<60):
            coords.append([b[0]])
for i in range(len(coords)):
    if (i+1>len(coords)-1) or i+2 > len(coords)-1:
        break
    else:
        coords1 = coords[i][0][:2]
        coords2 = coords[i+1][0][:2]
        coords3 = coords[i+2][0][:2]

    x,y = find_centre(coords1,coords2,coords3, filename)
    centre.append([x,y])

with open(filename, mode='r') as f:
    lines = csv.reader(f)
    for line in lines:
        if 'mx1' not in line:
            midpoint.append(line[:4])
            linear.append(line[4:])

        
x = []
y = []
Centre = []
for a in centre:
    if str(a[0]) != 'nan' and str(a[1]) != 'nan':
        x.append(round(a[0]))
        y.append(round(a[1]))

print(f'mode of centre is:({statistics.mode(x)},{statistics.mode(y)})')


# to view the data in Napari
viewer = nap.Viewer()
binary_image = double_thresholding(rotation_increment_data,lower_threshold,upper_threshold)

viewer.add_image(rotation_increment_data[70:2086], name='Original Images', colormap='gist_earth')
viewer.add_image(binary_image[70:2086], name='Binary Images', colormap='gray')

for i, b in enumerate(coords):
    #print(b)
    blob = b[0]
    if len(b) > 0:
        viewer.add_points(blob[:2], size=blob[2]*.4, face_color='red', name=f'Blobs at frame {indices[i]}')

for i in range(len(midpoint)):
    mx1, my1, mx2, my2 = midpoint[i]
    m1, c1, m2, c2 = linear[i]
    m1 = round(float(m1), 2)
    m2 = round(float(m2), 2)
    c1 = round(float(c1), 2)
    c2 = round(float(c2), 2)

    cx , cy = centre[i]
    #line =np.array([[mx1,my1],[cx,cy]])
    #print(line)

    viewer.add_shapes(np.array([[mx2,my2],[cx,cy]]),opacity=1,edge_color='yellow',shape_type='line',name=f'pair{i+1}: y={m2}x+{c2}')
    viewer.add_shapes(np.array([[mx1,my1],[cx,cy]]),opacity=1,edge_color='yellow',shape_type='line',name=f'pair{i+1}: y={m1}x+{c1}')     

for i, a in enumerate(centre):
    #print(i)
    #print(a)
    if len(a) >0:
        viewer.add_points(a, size=5, face_color='blue', border_color='whitesmoke',symbol='x', opacity=0.7, name=f'Centre of Circle no:{i+1}')

nap.run()


# barchart to visualise the mode

# Cx = []
# Cy = []
# numx = []
# numy = []
# for i in x:
#     if i not in Cx:
#         Cx.append(int(i))
#         numx.append(x.count(i))
# for i in y:
#     if i not in Cy:
#         Cy.append(int(i))
#         numy.append(y.count(i))
# print(Cx)
# print(Cy)
# print(x)

# print(y)


# fig, axs = plt.subplots(2, figsize=(10, 10))

# axs[0].bar(Cx, numx, color='b')
# axs[0].set_title('Barchart for X') 
# axs[0].set_ylabel('Values')

# axs[1].bar(Cy, numy, color='r') 
# axs[1].set_title('Barchart for Y')
# axs[1].set_ylabel('Values')

# plt.tight_layout()

# plt.show()




#finding the optimal threholds

# fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# # Original image
# ax[0].imshow(rotation_increment_data[0])
# ax[0].set_title('Original Image')
# ax[0].axis('off')

# # Binary image
# ax[1].imshow(binary_rotation_increment_data.astype(float)[0], cmap='gray')
# ax[1].set_title('Thresholded and Binarized Image')
# ax[1].axis('off')

# # Histogram of pixel values
# ax[2].hist(rotation_increment_data[0].ravel(), bins=256, color='k')
# ax[2].axvline(lower_threshold, color='r', linestyle='--', linewidth=1)
# ax[2].axvline(upper_threshold, color='r', linestyle='--', linewidth=1)
# ax[2].set_title('Histogram of Pixel Values')

# plt.tight_layout()
# plt.show()

