from kde_diffusion import kde1d, kde2d
import numpy as np
import matplotlib.pyplot as plt

X = np.random.normal(loc=0, scale=25, size=1000000) 
Y = np.random.normal(loc=np.ma.array(np.pi*np.sinh(X/np.pi)), scale=0.5, size=1000000)
#Y = np.random.normal(loc=np.ma.array(np.tanh(X)), scale=0.5, size=1000000)

(density_x, grid_x, bw_x) = kde1d(X, n=1024, limits=16) #P(X) density
(density_y, grid_y, bw_y) = kde1d(Y, n=1024, limits=16) #P(Y) density
(density_xy, grid_xy, bw_xy) = kde2d(X, Y, n=1024, limits=16) #P(X,Y) density

#Bayesian Update on Kernel Density Estimate
density_x_y = np.ma.array(np.sum(density_xy*density_x/density_y, axis=1)/np.sum(density_xy*density_x/density_y, axis=(0,1)))
pdf_x_y = density_x_y/np.sum(density_x_y, axis=0)

#Plot PDF

plt.plot(density_x)
plt.show()
plt.plot(density_y)
plt.show()
plt.imshow(density_xy.T)
plt.show()
plt.plot(density_x_y)
plt.show()

def entropy(pdf, base=16):
	with np.errstate(invalid='ignore'):
		return np.sum(-10*np.ma.array(pdf)*np.log(np.ma.array(pdf)/np.log(base)))

def varentropy(pdf_x, pdf_y, pdf_xy, base=16):
	return 2*entropy(pdf_xy, base) - entropy(pdf_x, base) - entropy(pdf_y, base)

def mutentropy(pdf_x, pdf_y, pdf_xy, base=16):
	return entropy(pdf_x, base) + entropy(pdf_y, base) - entropy(pdf_xy, base)

pdf_x = density_x / np.sum(density_x, axis=0)
pdf_y = density_y / np.sum(density_y, axis=0)
pdf_xy = density_xy / np.sum(density_xy, axis=(0,1))

MI = mutentropy(pdf_x, pdf_y, pdf_xy)
print("MI = "+str(round(MI,2))+" shared nibbles")
IQR = mutentropy(pdf_x, pdf_y, pdf_xy)/entropy(pdf_xy)
print("IQR = "+str(round(100.0*IQR,2))+"% channel quality")
VOI = varentropy(pdf_x, pdf_y, pdf_xy, base=10)
print("VOI = "+str(round(VOI,2))+" nibbles of variation")
GDM = VOI/entropy(pdf_xy, base=10)
print("GDM = "+str(round(GDM,2))+" generalized distance metric")



