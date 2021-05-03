import numpy as np
import matplotlib.pyplot as plt

phi = (lambda x,m,s: (1.0/np.sqrt(2.0*np.pi*s*s))*np.exp(-(x-m)**2/(2.0*s*s)))
f_vu = (lambda V,U,Dv,Du: 1.0/(Dv*Du))
f_xy = (lambda vx,vy,Dv,Du: (1.0/(Dv*Du))*((vx*vx+vy*vy)**(1.5)+vy*vy)/((vx*vx+vy*vy)**2*((vy/vx)**2+1)))
f_x_vu = (lambda vx,v,u,s: phi(vx,v*np.cos(u),s))
f_y_vu = (lambda vy,v,u,s,t: phi(vy,v*np.sin(u)-9.8*t,s))

f_vu_xy = (lambda vx,vy,v,u,sx,sy,t,Dv,Du: f_x_vu(vx,v,u,sx)*f_y_vu(vy,v,u,sy,t)*f_vu(v,u,Dv,Du)/f_xy(vx,vy,Dv,Du))

N_points = 256
V, U = np.mgrid[1:100:N_points*1j, 0:np.pi/2.0:N_points*1j]
Vx, Vy = np.mgrid[1:100.0/np.sqrt(2):N_points*1j, 1:100.0/np.sqrt(2):N_points*1j]

"""
plt.imshow(f_xy(Vx,Vy,95.0,5*np.pi/12.0), cmap=plt.cm.gist_heat, extent=[0.0,100.0/np.sqrt(2.0),0.0,100.0/np.sqrt(2.0)], aspect="auto")
plt.xlabel("U: Inclination angle (radians)")
plt.title("f(vx,vy)")
plt.show()"""

# Input reported velocities from Vx and Vy sensors
vx = 40
vy = 20

plt.imshow(f_vu_xy(vx,vy,V,U,3,3,0,99,np.pi/2.0).T, origin='lower', cmap=plt.cm.gist_heat, extent=[1,100,0,90], aspect="auto", interpolation='spline36')
plt.xlabel("V: Muzzle speed (meters/sec)")
plt.ylabel("U: Inclination angle (degrees)")
plt.title("f(v,u|vx,vy)")
plt.show()
