import numpy as np
import math
import matplotlib.pyplot as plt

# The map specific functions
phi_1 = -math.pi/2
lambdda_0 = 0

def transform(phi, lambdda):
    c = np.arccos(math.sin(phi_1)*math.sin(phi)+math.cos(phi_1)*math.cos(phi)*math.cos(lambdda-lambdda_0))
    k = c/math.sin(c)
    x =k*math.cos(phi)*math.sin(lambdda-lambdda_0)
    y= k*(math.cos(phi_1)*math.sin(phi) - math.sin(phi_1)*math.cos(phi)*math.cos(lambdda))
    return (y,x)

def get_scale_factors_at(phi, lambdda):
    (y_phi_h,x_phi_h) = transform(phi + h, lambdda)
    (y_phi_minush,x_phi_minush) = transform(phi - h, lambdda)
    
    (y_l_h,x_l_h) = transform(phi, lambdda + h)
    (y_l_minush,x_l_minush) = transform(phi, lambdda - h)
    
    x_phi = ( x_phi_h - x_phi_minush ) / (2*h)
    x_lamb = (x_l_h - x_l_minush ) / (2*h)
    y_phi = ( y_phi_h - y_phi_minush ) / (2*h)
    y_lamb = (y_l_h - y_l_minush) / (2*h)

    
    T = [[x_lamb/(math.cos(phi)), x_phi], [y_lamb/math.cos(phi), y_phi]]
    
    s = np.linalg.svd(T, compute_uv=False) # Computation of singular value decomposition (SVD)
    return s[0],s[1]

# The distortions

def I_score(num_points,points):
    total_array = np.zeros((num_points))
    for i in range(num_points):
        a,b = get_scale_factors_at(points[i,0], points[i,1])
        value = math.log(a/b)
        total_array[i]=value
    return rms(total_array)

def A_score(num_points,points):
    total_array = np.zeros((num_points))
    for i in range(num_points):
        a,b = get_scale_factors_at(points[i,0], points[i,1])
        value = math.log(a*b)
        total_array[i]= value
    return rms(total_array-np.mean(total_array))

def D_score(num_points):
    dist = np.zeros(num_points)
    for i in range(num_points):
        lon = np.random.uniform(- np.pi,  np.pi)
        lat = np.arcsin(np.random.uniform(-1,1))
                  
        lon2 = np.random.uniform(- np.pi,  np.pi)
        lat2 = np.arcsin(np.random.uniform(-1,1))

        x1 = [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        x2 = [np.cos(lat2) * np.cos(lon2), np.cos(lat2) * np.sin(lon2), np.sin(lat2)]
        dum = np.dot(x1, x2)
        dist_globe = np.arccos(dum)

        x1, y1 = transform(lat, lon)
        x2, y2 = transform(lat2, lon2)

        dist_map = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        dist[i] = dist_map / dist_globe
    idx = np.where(np.isfinite(dist) & (dist > 0))[0]
    return rms(np.log(dist[idx]) - np.mean(np.log(dist[idx])))

def rms(array):
    squared_array = np.square(array)
    mean_squared = np.mean(squared_array)
    rms_value = np.sqrt(mean_squared)
    return rms_value

# The function to plot the amount of points for the distortions
def A_plot(max_points):
    total_values = np.zeros((max_points//100))
    for i in range(max_points//100):
        lambdda = np.random.uniform(- np.pi, np.pi, i*100)
        phi = np.arcsin(np.random.uniform(-1,1,i*100))
        points = np.vstack((phi, lambdda)).T
        total_values[i] = A_score(i*100,points)
        
    x_values = np.arange(0, max_points, 100)

    mean_value = np.mean(total_values[1:])
    plt.axhline(mean_value, color='r', linestyle='--', label=f'Mean = {mean_value:.2f}')

    plt.plot(x_values, total_values, label='Area distortion')
    plt.xlabel('Amount of points generated')
    plt.ylabel('Area distortion')
    plt.title('Plot of Area distortion')
    plt.legend()
    plt.savefig("A_score_azimuthal", bbox_inches='tight')
    plt.show()

def I_plot(max_points):
    total_values = np.zeros((max_points//100))
    for i in range(max_points//100):
        lambdda = np.random.uniform(- np.pi, np.pi, i*100)
        phi = np.arcsin(np.random.uniform(-1,1,i*100))
        points = np.vstack((phi, lambdda)).T
        total_values[i] = I_score(i*100,points)
        
    x_values = np.arange(0, max_points, 100)

    mean_value = np.mean(total_values[1:])
    plt.axhline(mean_value, color='r', linestyle='--', label=f'Mean = {mean_value:.2f}')


    plt.plot(x_values, total_values, label='Isotropy distortion')
    plt.xlabel('Amount of points generated')
    plt.ylabel('Isotropy distortion')
    plt.title('Plot of Isotropy distortion')
    plt.legend()
    plt.savefig("I_score_azimuthal", bbox_inches='tight')
    plt.show()

def D_plot(max_points):
    total_values = np.zeros((max_points//100))
    for i in range(max_points//100):
        total_values[i] = D_score(i*100)
        
    x_values = np.arange(0, max_points, 100)

    mean_value = np.mean(total_values[1:])
    plt.axhline(mean_value, color='r', linestyle='--', label=f'Mean = {mean_value:.2f}')

    plt.plot(x_values, total_values, label='Distance distortion')
    plt.xlabel('Amount of points generated')
    plt.ylabel('Distance distortion')
    plt.title('Plot of distance distortion')
    plt.legend()
    plt.savefig("D_score_azimuthal", bbox_inches='tight')
    plt.show()
