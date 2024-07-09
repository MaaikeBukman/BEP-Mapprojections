import numpy as np
import math

h = 0.00001
phi_1 = -math.pi/2
lambdda_0 = 0

# Number of points to sample
num_points = 30000

# Generate random phi and lambda uniformly 
lambdda = np.random.uniform(- np.pi, np.pi, num_points)
phi = np.arcsin(np.random.uniform(-1,1,num_points))

# Combine phi, lambda into a single array of shape (num_points, 2)
points = np.vstack((phi, lambdda)).T

# The map specific functions
def transform(phi, lambdda):
    c = np.arccos(math.sin(phi_1)*math.sin(phi)+math.cos(phi_1)*math.cos(phi)*math.cos(lambdda-lambdda_0))
    k = c/math.sin(c)
    x =k*math.cos(phi)*math.sin(lambdda-lambdda_0)
    y= k*(math.cos(phi_1)*math.sin(phi) - math.sin(phi_1)*math.cos(phi)*math.cos(lambdda))
    return (y,x)

def get_scale_factors_at(phi, lambdda):
    x_phi = math.cos(lambdda)
    x_lamb = - (math.pi/2 + phi)*math.sin(lambdda)
    y_phi = math.sin(lambdda)
    y_lamb = (math.pi/2 + phi)*math.cos(lambdda)
    
    T = [[x_lamb/(math.cos(phi)), x_phi], [y_lamb/math.cos(phi), y_phi]]
    
    s = np.linalg.svd(T, compute_uv=False) # Computation of singular value decomposition (SVD)
    return s[0],s[1]

def derivatives(phi,lambdda,alpha):
    x_phi = math.cos(lambdda)
    x_lamb = - (math.pi/2 + phi)*math.sin(lambdda)
    x_phi_phi = 0
    x_lamb_lamb = - (math.pi/2 + phi)*math.cos(lambdda) 
    x_phi_lamb = -math.sin(lambdda)
    H_x = np.array([[x_lamb_lamb, x_phi_lamb],[x_phi_lamb, x_phi_phi]])

    y_phi = math.sin(lambdda)
    y_lamb = (math.pi/2 + phi)*math.cos(lambdda)
    y_phi_phi = 0
    y_lamb_lamb = - (math.pi/2 + phi)*math.sin(lambdda)
    y_phi_lamb =  math.cos(lambdda)
    H_y = np.array([[y_lamb_lamb, y_phi_lamb],[y_phi_lamb, y_phi_phi]])
    J = np.array([[x_lamb, x_phi], [y_lamb, y_phi]])
    K = np.array([[1/(math.cos(phi)), 0], [0, 1]])

    u = np.dot(K,np.array([[math.cos(alpha)],[math.sin(alpha)]]))
    w = np.dot(K,np.array([[math.sin(2*alpha)],[-math.cos(alpha)**2]]))*math.tan(phi)

    vel = np.dot(J,u) 
    acc1 = np.dot(J,w)
    acc2 = np.array([[np.dot(u.T,np.dot(H_x,u))],[np.dot(u.T,np.dot(H_y,u))]])
    acc = np.array([[acc1[0,0]+acc2[0,0,0]],[acc1[1,0]+acc2[1,0,0]]])

    acc = np.array([[acc[0,0,0]],[acc[1,0,0]]])
    return vel,acc

# The function to determine the distortions
def I_score():
    total_array = np.zeros((num_points))
    for i in range(num_points):
        a,b = get_scale_factors_at(points[i,0], points[i,1])
        value = math.log(a/b)
        total_array[i]=value
    return rms(total_array)

def A_score():
    total_array = np.zeros((num_points))
    for i in range(num_points):
        a,b = get_scale_factors_at(points[i,0], points[i,1])
        value = math.log(a*b)
        total_array[i]= value
    return rms(total_array-np.mean(total_array))

def f_score(vel,acc):
    return abs(vel[0]*acc[1]-vel[1]*acc[0])/(np.dot(vel.T,vel))

def s_score(vel,acc):
    return abs(np.dot(vel.T,acc))/(np.dot(vel.T,vel))

def FS_score():
    n=100
    j=0
    alphas = np.linspace(0, 2*math.pi, n)
    total_f = np.zeros((num_points*n))
    total_s = np.zeros((num_points*n))
    for i in range(num_points):
        for alpha in alphas:
            vel,acc = derivatives(points[i,0], points[i,1] ,alpha)
            f = f_score(vel,acc)
            s = s_score(vel,acc)
            total_f[j]= f 
            total_s[j] =s 
            j=j+1
    return np.mean(total_f),np.mean(total_s)

def D_score():
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
