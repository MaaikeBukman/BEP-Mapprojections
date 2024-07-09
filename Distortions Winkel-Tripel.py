import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate

# Number of points to sample
num_points = 60000

# Generate random phi and lambda uniformly 
lambdda = np.random.uniform(- np.pi, np.pi, num_points)
phi = np.arcsin(np.random.uniform(-1,1,num_points))

# Combine phi, lambda into a single array of shape (num_points, 2)
points = np.vstack((phi, lambdda)).T

# The map specific functions
def transform(phi, lambdda):
    x = 0.5*(2* np.arccos(math.cos(phi)*math.cos(lambdda/2))/math.sqrt(1-math.cos(phi)**2*math.cos(lambdda/2)**2)*math.cos(phi)*math.sin(lambdda/2)+lambdda)
    y = 0.5*(2* np.arccos(math.cos(phi)*math.cos(lambdda/2))/math.sqrt(1-math.cos(phi)**2*math.cos(lambdda/2)**2)*math.sin(phi)+phi)
    return (y,x)

def get_scale_factors_at(phi, lambdda):
    D = np.arccos(math.cos(phi)*math.cos(lambdda/2))
    C = 1 - math.cos(phi)**2 * math.cos(lambdda/2)**2
    x_phi = (math.sin(lambdda/2)* math.sin(phi)*(D-math.cos(lambdda/2)*math.cos(phi)*math.sqrt(C)))/(math.sqrt(C)*-C)
    x_lamb = (1/2)*((2*math.cos(phi)**2*math.cos(lambdda/2)**2-math.cos(phi)**2-1)/-C+((math.cos(phi)**3-math.cos(phi))*D*math.cos(lambdda/2))/(math.sqrt(C)*-C))
    y_phi = ((math.cos(lambdda/2)+1)*((math.cos(lambdda/2)-1)*math.cos(phi)*D+(math.cos(lambdda/2)*math.cos(phi)**2-1)*math.sqrt(C)))/(2*math.sqrt(C)*-C)
    y_lamb = (math.cos(phi)*math.sin(phi)*math.sin(lambdda/2))/(4*C)-(math.cos(phi)**2*math.sin(phi)*D*math.cos(lambdda/2)*math.sin(lambdda/2))/(4*C**(3/2))
          
    T = [[x_lamb/(math.cos(phi)), x_phi], [y_lamb/math.cos(phi), y_phi]]
    
    s = np.linalg.svd(T, compute_uv=False) # Computation of singular value decomposition (SVD)
    return s[0],s[1]

def derivatives(phi,lambdda,alpha):
    C = 1 - math.cos(phi)**2 * math.cos(lambdda/2)**2
    D = np.arccos(math.cos(phi)*math.cos(lambdda/2))
    E = math.cos(phi)**2 * math.cos(lambdda) + math.cos(phi)**2 - 2
    p=phi
    l=lambdda
    x_phi = (math.sin(lambdda/2)* math.sin(phi)*(D-math.cos(lambdda/2)*math.cos(phi)*math.sqrt(C)))/(math.sqrt(C)*-C)
    x_lamb = (1/2)*((2*math.cos(phi)**2*math.cos(lambdda/2)**2-math.cos(phi)**2-1)/-C+((math.cos(phi)**3-math.cos(phi))*D*math.cos(lambdda/2))/(math.sqrt(C)*-C))
    x_phi_phi = -(((4 * math.cos(p) ** 3 - 6 * math.cos(p)) * math.cos(l / 2) ** 2 + 2 * math.cos(p)) * D + math.cos(l / 2) * (math.cos(p) ** 2 * math.cos(l / 2) ** 2 - 3 * math.cos(p) **2 + 2) * math.sqrt(-2*E)) * (-2*E) ** (-0.1e1 / 0.2e1) * math.sin(l / 2) / (math.cos(p) ** 4 * math.cos(l / 2) ** 4 - 2 * math.cos(p) ** 2 * math.cos(l / 2) ** 2 + 1)
    x_lamb_lamb = (4*math.cos(phi)**2*math.cos(lambdda/2)**2*D-3*math.cos(phi)*math.cos(lambdda/2)*math.sqrt(-2*E)+2*D)*math.sqrt(-2*E)*math.sin(phi)**2*math.sin(lambdda/2)*math.cos(phi)/(E*(math.cos(phi)**4*math.cos(lambdda/2)**4-2*math.cos(phi)**2*math.cos(lambdda/2)**2+1)*8)
    x_phi_lamb = -(-2*E) ** (-0.1e1 / 0.2e1) * (((math.cos(p) ** 3 - 3 * math.cos(p)) * math.cos(l / 2) ** 2 + 2 * math.cos(p)) * math.sqrt(-2*E) + 4 * (math.cos(p) ** 2 * math.cos(l / 2) ** 2 - 0.3e1 / 0.2e1 * math.cos(p) ** 2 + 0.1e1 / 0.2e1) * math.cos(l / 2) *D) * math.sin(p) / (2 * math.cos(p) ** 4 * math.cos(l / 2) ** 4 - 4 * math.cos(p) ** 2 * math.cos(l / 2) ** 2 + 2)
    H_x = np.array([[x_lamb_lamb, x_phi_lamb],[x_phi_lamb, x_phi_phi]])

    y_phi = ((math.cos(lambdda/2)+1)*((math.cos(lambdda/2)-1)*math.cos(phi)*D+(math.cos(lambdda/2)*math.cos(phi)**2-1)*math.sqrt(C)))/(2*math.sqrt(C)*-C)
    y_lamb = (math.cos(phi)*math.sin(phi)*math.sin(lambdda/2))/(4*C)-(math.cos(phi)**2*math.sin(phi)*D*math.cos(lambdda/2)*math.sin(lambdda/2))/(4*C**(3/2))
    y_phi_phi = (-4 * math.cos(p) ** 2 * math.cos(l / 2) ** 2 * D + 3 * math.cos(p) * math.cos(l / 2) * math.sqrt(-2*E) - 2 * D) * math.sin(l / 2) ** 2 * math.sin(p) * (-2*E) ** (-0.1e1 / 0.2e1) / (2 * math.cos(p) ** 4 * math.cos(l / 2) ** 4 - 4 * math.cos(p) ** 2 * math.cos(l / 2) ** 2 + 2)
    y_lamb_lamb = 2 * ((-math.cos(l / 2) ** 4 * math.cos(p) ** 3 - 2 * math.cos(p) * math.cos(l / 2) ** 2 * math.sin(p) ** 2 + math.cos(p)) * D + math.sqrt(-2*E) * (math.cos(p) ** 2 * math.cos(l / 2) ** 2 - 0.3e1 / 0.2e1 * math.cos(p) ** 2 + 0.1e1 / 0.2e1) * math.cos(l / 2)) * math.cos(p) * (4 + (-2 * math.cos(l) - 2) * math.cos(p) ** 2) ** (-0.1e1 / 0.2e1) * math.sin(p) / (8 * math.cos(p) ** 4 * math.cos(l / 2) ** 4 - 16 * math.cos(p) ** 2 * math.cos(l / 2) ** 2 + 8)
    y_phi_lamb = ((math.cos(p) ** 4 * math.cos(l / 2) ** 2 + 2 * math.cos(p) ** 2 * math.sin(l / 2) ** 2 - 1) * math.sqrt(4 + (-2 * math.cos(l) - 2) * math.cos(p) ** 2) + 2 * math.acos(math.cos(p) * math.cos(l / 2)) * math.cos(p) * math.cos(l / 2) * (math.cos(p) ** 2 * math.cos(l / 2) ** 2 - 3 * math.cos(p) ** 2 + 2)) * (4 + (-2 * math.cos(l) - 2) * math.cos(p) ** 2) ** (-0.1e1 / 0.2e1) * math.sin(l / 2) / (4 * math.cos(p) ** 4 * math.cos(l / 2) ** 4 - 8 * math.cos(p) ** 2 * math.cos(l / 2) ** 2 + 4)
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

##def FS_int_score():
##    f = lambda alpha, phi, lambdda: f_score(derivatives(phi,lambdda,alpha)[0],derivatives(phi,lambdda,alpha)[1])*math.cos(phi)
##    x = integrate.tplquad(f,-math.pi,math.pi,-math.pi/2, math.pi/2, 0, 2*math.pi)[0]
##    total_f = x/(8*math.pi**2)
##    s = lambda alpha, phi, lambdda: s_score(derivatives(phi,lambdda,alpha)[0],derivatives(phi,lambdda,alpha)[1])*math.cos(phi)
##    x = integrate.tplquad(s,-math.pi,math.pi,-math.pi/2, math.pi/2, 0, 2*math.pi)[0]
##    total_s = x/(8*math.pi**2)
##    return total_f

def F_score():
    n=100
    j=0
    alphas = np.linspace(0, 2*math.pi, n)
    total_f = np.zeros((num_points*n))
    for i in range(num_points):
        for alpha in alphas:
            xdot,xddot,ydot,yddot = derivatives(points[i,0], points[i,1],alpha)
            f = f_score(xdot,xddot,ydot,yddot)
            total_f[j]=f
            j=j+1
    return np.mean(abs(total_f))

def S_score():
    n=100
    j=0
    alphas = np.linspace(0, 2*math.pi, n)
    total_s = np.zeros((num_points*n))
    for i in range(num_points):
        for alpha in alphas:
            xdot,xddot,ydot,yddot = derivatives(points[i,0], points[i,1],alpha)
            s = s_score(xdot,xddot,ydot,yddot)
            total_s[j]=s
            j=j+1
    return np.mean(abs(total_s)),total_s

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
