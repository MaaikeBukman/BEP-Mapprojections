import matplotlib.image as image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import math
from scipy.ndimage import map_coordinates

# The map specific functions
def transform(phi, lambdda):
    x = lambdda
    y = math.log(math.tan(phi/2+math.pi/4))
    return (y,x)

map_width = 2000
map_length = 2000

x_domain = math.pi
y_domain = math.pi

# 

def x_to_pixel(x):
    return round((x/( 2 *(x_domain)) + 0.5)*map_width)

def pixel_to_x(pixel):
    return (pixel/map_width - 0.5) * 2 *(x_domain)

def y_to_pixel(y):
    return round((y/(2*y_domain) + 0.5)*map_length)

def pixel_to_y(pixel):
    return (pixel/map_length - 0.5) *(2*y_domain)

def phi_to_pixel(phi):
    return round((phi + 0.5*math.pi)/math.pi*height)

def lambdda_to_pixel(lambdda):
    return round((lambdda + math.pi)/(2*math.pi)*width)

def pixel_to_phi(pixel):
    return (pixel*math.pi)/height-0.5*math.pi

def pixel_to_lambdda(pixel):
    return (pixel*2*math.pi)/width-math.pi
 
def plot_parallels():
    parallels = np.linspace(-0.4999*math.pi, 0.4999*math.pi, num=9) #These have a certain lattitude phi, while longitude changes
    lambddas = np.linspace(-math.pi, math.pi, map_width)
    xy = np.empty((9, map_width, 2))
    j=0
    for parallel in parallels:
        i = 0
        for lambdda in lambddas:
            (y,x) = transform(parallel, lambdda)
            adjusted = (y_to_pixel(y), x_to_pixel(x))
            xy[j][i] = adjusted
            i = i + 1
        j = j+1
        
    for j in range(0, len(parallels)):
        plt.plot(xy[j,:,1], xy[j,:,0], "w", alpha=0.3, linewidth=0.5)

def plot_meridians():
    meridians = np.linspace(-math.pi, math.pi, num=18)  # These have a certain longitude lambdda, while latitude changes
    phis = np.linspace(-0.4999 * math.pi, 0.4999 * math.pi, map_length)
    xy_meridians = np.empty((18, map_length, 2))
    j = 0
    for meridian in meridians:
        i = 0
        for phi in phis:
            (y, x) = transform(phi, meridian)
            adjusted = (y_to_pixel(y), x_to_pixel(x))
            xy_meridians[j][i] = adjusted
            i = i + 1

        j = j + 1

    for j in range(0, len(meridians)):
        plt.plot(xy_meridians[j, :, 1], xy_meridians[j, :, 0], "w", alpha=0.3,linewidth=0.5)

# world is the input map and new_world is the output map
new_world = np.full(  ( map_length , map_width , 3 ), 255, dtype=np.uint8 ) # this will be (y,x)
world = image.imread("4_no_ice_clouds_mts_8k.jpg") # longitude theta by latitude phi, in this map

# Define the dimensions of the image
height, width, _ = world.shape

# Perform the transformation without the inverse
phis = np.linspace(-0.4999* math.pi, 0.4999* math.pi, 2*map_length)
lambddas = np.linspace(-0.999*math.pi, 0.999* math.pi, map_width)
for phi in phis:
    for lambdda in lambddas:
        (y,x) = transform(phi,lambdda)
        (phi_pixel, lambdda_pixel) =  (phi_to_pixel(phi),lambdda_to_pixel(lambdda))
        if (phi_pixel >= 0 and lambdda_pixel >=0 and phi_pixel < height and lambdda_pixel < width):  # Check boundary conditions
            pixel_value = world[phi_pixel][lambdda_pixel]
            (x_pixel,y_pixel) = (x_to_pixel(x),y_to_pixel(y))
            if (y_pixel >= 0 and x_pixel >=0 and y_pixel < map_length and x_pixel < map_width): 
                new_world[y_pixel][x_pixel] = pixel_value 

### Interpolation to fill black pixels
##black_pixels = np.where((new_world == 0).all(axis=2))
##
##for y_pixel, x_pixel in zip(*black_pixels):
##    non_black_neighbors = []
##    for dy in [-1, 0, 1]:
##        for dx in [-1, 0, 1]:
##            ny, nx = y_pixel + dy, x_pixel + dx
##            if 0 <= ny < map_length and 0 <= nx < map_width and not (dy == 0 and dx == 0):
##                neighbor_value = new_world[ny, nx]
##                if not (neighbor_value == 0).all():
##                    non_black_neighbors.append(neighbor_value)
##    
##    if non_black_neighbors:
##        new_world[y_pixel, x_pixel] = np.mean(non_black_neighbors, axis=0)
        
plt.figure(dpi=400)
plt.axis('off')
plot_parallels()
plot_meridians()
plt.imshow(new_world)
plt.savefig("no_inv_map_plot_Mercator.png", bbox_inches='tight')

