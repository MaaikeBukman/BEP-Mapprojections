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
    divide = math.sin(np.arccos(math.cos(2*phi/3)*math.cos(lambdda/(2*math.sqrt(3)))))
    x = -(2*np.arccos(math.cos(2*phi/3)*math.cos(lambdda/(2*math.sqrt(3))))*math.cos(2*phi/3)*math.sin(lambdda/(2*math.sqrt(3))))/divide
    y = (np.arccos(math.cos(2*phi/3)*math.cos(lambdda/(2*math.sqrt(3))))*math.sin(2*phi/3))/divide
    return (y,x)

def get_scale_factors_at(phi, lambdda):
    z = np.arccos(math.cos(2*phi/3)*math.cos(lambdda/(2*math.sqrt(3))))
    w = -math.cos(2*phi/3)**2*math.cos(lambdda/(2*math.sqrt(3)))**2+1
    x_phi = (4*(math.cos(lambdda*math.sqrt(3)/6)*math.cos(2*phi/3)*math.sqrt(w)-z)*math.sin(lambdda*math.sqrt(3)/6)*math.sin(2*phi/3))/(3*w**(3/2))
    x_lamb = ((math.cos(2*phi/3)*math.sin(lambdda*math.sqrt(3)/6)**2*math.sqrt(w)+math.sin(2*phi/3)**2*z*math.cos(lambdda*math.sqrt(3)/6))*math.sqrt(3)*math.cos(2*phi/3))/(3*w**(3/2))
    y_phi = (2*(math.sin(lambdda*math.sqrt(3)/6)**2*z*math.cos(2*phi/3)+math.sin(2*phi/3)**2*math.cos(lambdda*math.sqrt(3)/6)*math.sqrt(w)))/(3*w**(3/2))
    y_lamb = 1/6*(math.sqrt(3)*math.sin(lambdda*math.sqrt(3)/6)*math.sin(2*phi/3)*(math.cos(2*phi/3)/w-(z*math.cos(2*phi/3)**2*math.cos(lambdda*math.sqrt(3)/6))/(w**(3/2))))
          
    T = [[x_lamb/(math.cos(phi)), x_phi], [y_lamb/math.cos(phi), y_phi]]
    
    s = np.linalg.svd(T, compute_uv=False) # Computation of singular value decomposition (SVD)
    return s[0],s[1]

def derivatives(phi,lambdda,alpha):
    p=phi
    l=lambdda
    z = np.arccos(math.cos(2*phi/3)*math.cos(lambdda/(2*math.sqrt(3))))
    w = -math.cos(2*phi/3)**2*math.cos(lambdda/(2*math.sqrt(3)))**2+1
    
    x_phi = (4*(math.cos(lambdda*math.sqrt(3)/6)*math.cos(2*phi/3)*math.sqrt(w)-z)*math.sin(lambdda*math.sqrt(3)/6)*math.sin(2*phi/3))/(3*w**(3/2))
    x_lamb = ((math.cos(2*phi/3)*math.sin(lambdda*math.sqrt(3)/6)**2*math.sqrt(w)+math.sin(2*phi/3)**2*z*math.cos(lambdda*math.sqrt(3)/6))*math.sqrt(3)*math.cos(2*phi/3))/(3*w**(3/2))
    x_phi_phi = -16 * math.sin(l * math.sqrt(3) / 6) * (math.cos(l * math.sqrt(3) / 6) * (math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2 - 3 * math.cos(0.2e1 / 0.3e1 * p) ** 2 + 2) * math.sqrt(1 - math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2) / 2 + math.acos(math.cos(0.2e1 / 0.3e1 * p) * math.cos(l * math.sqrt(3) / 6)) * (0.1e1 / 0.2e1 + (math.cos(0.2e1 / 0.3e1 * p) ** 2 - 0.3e1 / 0.2e1) * math.cos(l * math.sqrt(3) / 6) ** 2) * math.cos(0.2e1 / 0.3e1 * p)) * (1 - math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2) ** (-0.1e1 / 0.2e1) / (9 * math.cos(0.2e1 / 0.3e1 * p) ** 4 * math.cos(l * math.sqrt(3) / 6) ** 4 - 18 * math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2 + 9)
    x_lamb_lamb = math.sin(0.2e1 / 0.3e1 * p) ** 2 * (-2 * math.acos(math.cos(0.2e1 / 0.3e1 * p) * math.cos(l * math.sqrt(3) / 6)) * math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2 + 3 * math.cos(l * math.sqrt(3) / 6) * math.cos(0.2e1 / 0.3e1 * p) * math.sqrt(1 - math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2) - math.acos(math.cos(0.2e1 / 0.3e1 * p) * math.cos(l * math.sqrt(3) / 6))) * math.sin(l * math.sqrt(3) / 6) * math.cos(0.2e1 / 0.3e1 * p) * (1 - math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2) ** (-0.1e1 / 0.2e1) / (6 * math.cos(0.2e1 / 0.3e1 * p) ** 4 * math.cos(l * math.sqrt(3) / 6) ** 4 - 12 * math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2 + 6)
    x_phi_lamb = -2 * (((math.cos(0.2e1 / 0.3e1 * p) ** 3 - 3 * math.cos(0.2e1 / 0.3e1 * p)) * math.cos(l * math.sqrt(3) / 6) ** 2 + 2 * math.cos(0.2e1 / 0.3e1 * p)) * math.sqrt(1 - math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2) + 2 * (math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2 - 0.3e1 / 0.2e1 * math.cos(0.2e1 / 0.3e1 * p) ** 2 + 0.1e1 / 0.2e1) * math.acos(math.cos(0.2e1 / 0.3e1 * p) * math.cos(l * math.sqrt(3) / 6)) * math.cos(l * math.sqrt(3) / 6)) * math.sqrt(3) * (1 - math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2) ** (-0.1e1 / 0.2e1) * math.sin(0.2e1 / 0.3e1 * p) / (9 * math.cos(0.2e1 / 0.3e1 * p) ** 4 * math.cos(l * math.sqrt(3) / 6) ** 4 - 18 * math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2 + 9)
    H_x = np.array([[x_lamb_lamb, x_phi_lamb],[x_phi_lamb, x_phi_phi]])

    y_phi = (2*(math.sin(lambdda*math.sqrt(3)/6)**2*z*math.cos(2*phi/3)+math.sin(2*phi/3)**2*math.cos(lambdda*math.sqrt(3)/6)*math.sqrt(w)))/(3*w**(3/2))
    y_lamb = 1/6*(math.sqrt(3)*math.sin(lambdda*math.sqrt(3)/6)*math.sin(2*phi/3)*(math.cos(2*phi/3)/w-(z*math.cos(2*phi/3)**2*math.cos(lambdda*math.sqrt(3)/6))/(w**(3/2))))
    y_phi_phi = 4 * math.sin(l * math.sqrt(3) / 6) ** 2 * (-2 * math.acos(math.cos(0.2e1 / 0.3e1 * p) * math.cos(l * math.sqrt(3) / 6)) * math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2 + 3 * math.cos(l * math.sqrt(3) / 6) * math.cos(0.2e1 / 0.3e1 * p) * math.sqrt(1 - math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2) - math.acos(math.cos(0.2e1 / 0.3e1 * p) * math.cos(l * math.sqrt(3) / 6))) * math.sin(0.2e1 / 0.3e1 * p) * (1 - math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2) ** (-0.1e1 / 0.2e1) / (9 * math.cos(0.2e1 / 0.3e1 * p) ** 4 * math.cos(l * math.sqrt(3) / 6) ** 4 - 18 * math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2 + 9)
    y_lamb_lamb = (-math.acos(math.cos(0.2e1 / 0.3e1 * p) * math.cos(l * math.sqrt(3) / 6)) * math.cos(0.2e1 / 0.3e1 * p) * (math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 4 + 2 * math.sin(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2 - 1) + math.sqrt(1 - math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2) * math.cos(l * math.sqrt(3) / 6) * (2 * math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2 - 3 * math.cos(0.2e1 / 0.3e1 * p) ** 2 + 1)) * math.cos(0.2e1 / 0.3e1 * p) * (1 - math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2) ** (-0.5e1 / 0.2e1) * math.sin(0.2e1 / 0.3e1 * p) / 12
    y_phi_lamb = math.sqrt(3) * math.sin(l * math.sqrt(3) / 6) * ((math.cos(0.2e1 / 0.3e1 * p) ** 4 * math.cos(l * math.sqrt(3) / 6) ** 2 + 2 * math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.sin(l * math.sqrt(3) / 6) ** 2 - 1) * math.sqrt(1 - math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2) + math.acos(math.cos(0.2e1 / 0.3e1 * p) * math.cos(l * math.sqrt(3) / 6)) * math.cos(0.2e1 / 0.3e1 * p) * math.cos(l * math.sqrt(3) / 6) * (math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2 - 3 * math.cos(0.2e1 / 0.3e1 * p) ** 2 + 2)) * (1 - math.cos(0.2e1 / 0.3e1 * p) ** 2 * math.cos(l * math.sqrt(3) / 6) ** 2) ** (-0.5e1 / 0.2e1) / 9
    H_y = np.array([[y_lamb_lamb, y_phi_lamb],[y_phi_lamb, y_phi_phi]])

    J = np.array([[x_lamb, x_phi], [y_lamb, y_phi]])
    K = np.array([[1/(math.cos(phi)), 0], [0, 1]])

    u = np.dot(K,np.array([[math.cos(alpha)],[math.sin(alpha)]]))
    u_perp = np.dot(K,np.array([[-math.sin(alpha)],[math.cos(alpha)]]))
    w = np.dot(K,np.array([[math.sin(2*alpha)],[-math.cos(alpha)**2]]))*math.tan(phi)
    #w = np.array([[2* math.tan(phi)*u[0,0]*u[1,0]],[-math.sin(phi)*math.cos(phi)*((u[0,0])**2)]])

    vel = np.dot(J,u) 
    vel_perp = np.dot(J,u_perp)
    #acc = np.dot(J,w)+np.array([[np.dot(u.T,np.dot(H_x,u))],[np.dot(u.T,np.dot(H_y,u))]])
    acc1 = np.dot(J,w)
    acc2 = np.array([[np.dot(u.T,np.dot(H_x,u))],[np.dot(u.T,np.dot(H_y,u))]])
    acc = np.array([[acc1[0,0]+acc2[0,0,0]],[acc1[1,0]+acc2[1,0,0]]])

    acc = np.array([[acc[0,0,0]],[acc[1,0,0]]])
    return vel,acc

# The function to determine the distortions
def I_score():
    j=0
    total_array = np.zeros((num_points))
    for i in range(num_points):
        a,b = get_scale_factors_at(points[i,0], points[i,1])
        value = math.log(a/b)
        total_array[i]=value
    return rms(total_array)

def A_score():
    j=0
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
    n=314
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
        #lon4 = points[i,0]
        #lon = np.pi * (2 * np.random.rand() - 1)
        lon = np.random.uniform(- np.pi,  np.pi)
        #lat = np.arcsin(np.random.rand() * (np.sin(math.pi/2) - np.sin(-math.pi/2)) + np.sin(-math.pi/2))
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
