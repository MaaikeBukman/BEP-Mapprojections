import matplotlib.image as image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import math
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from matplotlib import ticker

# The map specific functions
map_width = 2000
map_length = 1000

x_domain = math.pi
y_domain = math.pi/2

def transform(phi, lambdda):
    x = lambdda*math.cos(0)
    y = phi
    return (y,x)

def scale_factors(phi, lambdda):
    x_phi = 0
    x_lamb = 1
    y_phi = 1
    y_lamb = 0
        
    T = [[x_lamb/(math.cos(phi)), x_phi], [y_lamb/math.cos(phi), y_phi]]
    
    s = np.linalg.svd(T, compute_uv=False) # Computation of singular value decomposition (SVD)
    return s[0],s[1]

def derivatives(phi,lambdda,alpha):
    x_phi = 0
    x_lamb = 1
    x_phi_phi = 0 
    x_lamb_lamb = 0
    x_phi_lamb = 0 
    H_x = np.array([[x_lamb_lamb, x_phi_lamb],[x_phi_lamb, x_phi_phi]])
    
    y_phi = 1
    y_lamb = 0
    y_phi_phi = 0
    y_lamb_lamb = 0 
    y_phi_lamb = 0 
    H_y = np.array([[y_lamb_lamb, y_phi_lamb],[y_phi_lamb, y_phi_phi]])

    J = np.array([[x_lamb, x_phi], [y_lamb, y_phi]])
    K = np.array([[1/(math.cos(phi)), 0], [0, 1]])

    u = np.dot(K,np.array([[math.cos(alpha)],[math.sin(alpha)]]))
    u_perp = np.dot(K,np.array([[-math.sin(alpha)],[math.cos(alpha)]]))
    w = np.dot(K,np.array([[math.sin(2*alpha)],[-math.cos(alpha)**2]]))*math.tan(phi)

    vel = np.dot(J,u) 
    vel_perp = np.dot(J,u_perp)
    acc1 = np.dot(J,w)
    acc2 = np.array([[np.dot(u.T,np.dot(H_x,u))],[np.dot(u.T,np.dot(H_y,u))]])
    acc = np.array([[acc1[0,0]+acc2[0,0,0]],[acc1[1,0]+acc2[1,0,0]]])

    acc = np.array([[acc[0,0,0]],[acc[1,0,0]]])
    return vel,acc

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

# Number of points to sample
num_points = 30000

# Generate random phi and lambda uniformly 
lambdda = np.random.uniform(- np.pi, np.pi, num_points)
phi = np.arcsin(np.random.uniform(-1,1,num_points))

# Combine phi, lambda into a single array of shape (num_points, 2)
points = np.vstack((phi, lambdda)).T

# The global distortions
def I_score():
    j=0
    total_array = np.zeros((num_points))
    for i in range(num_points):
        a,b = scale_factors(points[i,0], points[i,1])
        value = math.log(a/b)
        total_array[i]=value
    return rms(total_array)

def A_score():
    j=0
    total_array = np.zeros((num_points))
    for i in range(num_points):
        a,b = scale_factors(points[i,0], points[i,1])
        value = math.log(a*b)
        total_array[i]= value
    return rms(total_array-np.mean(total_array)), np.mean(total_array)

def rms(array):
    squared_array = np.square(array)
    mean_squared = np.mean(squared_array)
    rms_value = np.sqrt(mean_squared)
    return rms_value

print("Area_distortion:",A_score()[0])
mean = A_score()[1]
print("Isotropy_distortion:",I_score())

# The local distortions
def i_score(phi,lambdda):
    a,b = scale_factors(phi, lambdda)
    value = math.log(a/b)
    return value

def a_score(phi, lambdda, mean):
    a,b = scale_factors(phi, lambdda)
    value = math.log(a*b)
    return value-mean

def f_score(vel,acc):
    return abs(vel[0]*acc[1]-vel[1]*acc[0])/(np.dot(vel.T,vel))

def s_score(vel,acc):
    return abs(np.dot(vel.T,acc))/(np.dot(vel.T,vel))

def fs_score(phi,lambdda):
    j=0
    n = 100
    alphas = np.linspace(0, 2*math.pi, n)
    total_f = np.zeros((n))
    total_s = np.zeros((n))
    for alpha in alphas:
        vel,acc = derivatives(phi, lambdda ,alpha)
        f = f_score(vel,acc)
        s = s_score(vel,acc)
        total_f[j]= f 
        total_s[j] =s
        j=j+1
    return (np.mean(total_f),np.mean(total_s))

# World is the input map and new_world is the output map
new_world = np.empty(  ( map_length , map_width , 3 ), dtype=np.uint8 ) # this will be (y,x)
world = image.imread("4_no_ice_clouds_mts_8k.jpg") # longitude theta by latitude phi, in this map
I_dist = np.empty(( map_length , map_width))
A_dist = np.empty((map_length,map_width))
F_dist = np.empty((map_length,map_width))
S_dist = np.empty((map_length,map_width))
y_pixels = np.linspace(1,map_length,map_length)
x_pixels = np.linspace(1,map_width,map_width)
X, Y = np.meshgrid(x_pixels, y_pixels)

# Define the dimensions of the image
height, width, _ = world.shape

# Perform the transformation
phis = np.linspace(-0.4999* math.pi, 0.4999* math.pi, map_length)
lambddas = np.linspace(-0.999*math.pi, 0.999* math.pi, map_width)
j=0
for phi in phis:
    for lambdda in lambddas:
        (y,x) = transform(phi,lambdda)
        (phi_pixel, lambdda_pixel) =  (phi_to_pixel(phi),lambdda_to_pixel(lambdda))
        if (phi_pixel >= 0 and lambdda_pixel >=0 and phi_pixel < height and lambdda_pixel < width):  # Check boundary conditions
            pixel_value = world[phi_pixel][lambdda_pixel]
            (x_pixel,y_pixel) = (x_to_pixel(x),y_to_pixel(y))
            if (y_pixel >= 0 and x_pixel >=0 and y_pixel < map_length and x_pixel < map_width):
                A_dist[y_pixel][x_pixel]= a_score(phi,lambdda,mean)
                I_dist[y_pixel][x_pixel]= i_score(phi,lambdda)
                (F_dist[y_pixel][x_pixel],S_dist[y_pixel][x_pixel]) = fs_score(phi,lambdda)
                new_world[y_pixel][x_pixel] = pixel_value 

# The levels and colors of the contourplot
alevels = [-1,-1e-1,-1e-2,-1e-3,0,1e-3, 1e-2, 1e-1, 1,10]
ilevels = [0,1e-3, 1e-2, 1e-1, 1,10]
acolors = [
    "#006400",  # Dark Green
    "#228B22",  # Forest Green
    "#3CB371",  # Medium Sea Green
    "#32CD32",  # Lime Green
    "#90EE90",  # Light Green
    "#87CEEB",  # Sky Blue
    "#00BFFF",  # Deep Sky Blue
    "#1E90FF",  # Dodger Blue
    "#4169E1",  # Royal Blue
    "#000080"   # Navy Blue
]
icolors = [
    "#90EE90",  # Light Green
    "#87CEEB",  # Sky Blue
    "#00BFFF",  # Deep Sky Blue
    "#1E90FF",  # Dodger Blue
    "#4169E1",  # Royal Blue
    "#000080"   # Navy Blue
]

# Contour plot for flexion distortion 
plt.figure(dpi=400)
plt.axis('off')
plot_parallels()
plot_meridians()
plt.imshow(new_world)

plt.contourf(X, Y, F_dist, levels=ilevels,colors=icolors, extend='neither',alpha = 0.5)
plt.colorbar(label='Flexion distortion',shrink=0.5)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.tight_layout() 
plt.savefig("Contourplot_flexion_equirectangular", bbox_inches='tight')
plt.show()

# Contour plot for skewness distortion
plt.figure(dpi=400)
plt.axis('off')
plot_parallels()
plot_meridians()
plt.imshow(new_world)

plt.contourf(X, Y, S_dist, levels=ilevels,colors=icolors, extend='neither',alpha = 0.5)
plt.colorbar(label='Skewness distortion',shrink=0.5)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.tight_layout() 
plt.savefig("Contourplot_skewness_equirectangular", bbox_inches='tight')
plt.show()

# Contour plot for area distortion 
plt.figure(dpi=400)
plt.axis('off')
plot_parallels()
plot_meridians()
plt.imshow(new_world)

plt.contourf(X, Y, A_dist, levels=alevels,colors=acolors, extend='neither',alpha = 0.5)
plt.colorbar(label='Area distortion',shrink=0.5)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.tight_layout() 
plt.savefig("Contourplot_Area_equirectangular", bbox_inches='tight')
plt.show()

# Contour plot for Isotropy distortion
plt.figure(dpi=400)
plt.axis('off')
plot_parallels()
plot_meridians()
plt.imshow(new_world)
                
plt.contourf(X, Y, I_dist, levels=ilevels, colors=icolors, extend='neither',alpha = 0.5)
plt.colorbar(label='Isotropy distortion',shrink=0.5)
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.tight_layout() 
plt.savefig("Contourplot_Isotropy_equirectangular", bbox_inches='tight')
plt.show()
