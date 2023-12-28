import matplotlib as mpl
mpl.rc("text",usetex=True)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import h5py

fs = 20

################################################################################################
# Function to create the Galactocentric profile of CRs
def func_CR_sol(CR_map, Rmax, NR):

    # Size of the CR map
    Nx, Ny = CR_map.shape

    # Create a grid of coordinates
    x, y = np.ogrid[:Nx, :Ny]
    Rkpc = np.linspace(0.0,2.0,NR) 
    print(Rkpc)

    # Define the parameters of the ring
    dx = 2.0*Rmax/(Ny-1)
    # center = (Nx // 2, Ny // 2)
    center = (Nx//2, int((7.9+Rmax)/dx))
    Rpix = Rkpc*center[0]/Rmax
    CR_pro = np.zeros([2*(NR-1),3])
    CR_pro_full = np.zeros([2*(NR-1),3])

    # Make the Galactocentric profile
    for i in range(NR-1):
        # Create a circular mask
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= Rpix[i+1] ** 2
        mask &= (x - center[0]) ** 2 + (y - center[1]) ** 2 >= Rpix[i] ** 2

        # Apply the mask to the image
        masked_image = np.copy(CR_map)
        masked_image[~mask] = np.nan  # Set pixels outside the ring to nan

        CR_pro_full[2*i,:]= np.nanmean(masked_image), np.nanmin(masked_image), np.nanmax(masked_image)
        CR_pro_full[2*i+1,:]= np.nanmean(masked_image), np.nanmin(masked_image), np.nanmax(masked_image)
        CR_pro[2*i,:]= np.nanpercentile(masked_image,50), np.nanpercentile(masked_image,5), np.nanpercentile(masked_image,95)
        CR_pro[2*i+1,:]= np.nanpercentile(masked_image,50), np.nanpercentile(masked_image,5), np.nanpercentile(masked_image,95)

    CR_pro *= 1.0e22 # 1.0e-13 GeV^-1 cm^-3
    CR_pro_full *= 1.0e22 # 1.0e-13 GeV^-1 cm^-3

    Rkpc_plot = np.sort(np.concatenate((Rkpc[0:-1], Rkpc[1:]-0.001), axis=0))

    return Rkpc_plot, CR_pro, CR_pro_full, masked_image

################################################################################################
# Read the CR map from the point source simulations
with h5py.File('nCR_30GeV_pts_CAB98.hdf5', 'r') as hf:
    CR_map_pts = np.array(hf['nCR eV^-1 cm^-3'])

Rkpc_plot_pts, CR_pro_pts, CR_pro_full_pts, masked_image = func_CR_sol(CR_map_pts,10.0,11)

################################################################################################
# Plot the CR profile
fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

# Rg_dif, nCR_dif, err_Rg_dif=np.loadtxt("data_nCR_30GeV_gamma_dif.dat",unpack=True,usecols=[0,1,2])
# Rg_mcl, nCR_mcl, err_nCR_mcl=np.loadtxt("data_nCR_30GeV_gamma_mcl.dat",unpack=True,usecols=[0,1,2])

ax.plot(Rkpc_plot_pts,CR_pro_pts[:,0],'r--',linewidth=3.0)
ax.fill_between(Rkpc_plot_pts,CR_pro_pts[:,1],CR_pro_pts[:,2],color=[(255.0/255.0,102.0/255.0,102.0/255.0)],label=r'${\rm 90\%\, Uncertainty}$')
ax.fill_between(Rkpc_plot_pts,CR_pro_full_pts[:,1],CR_pro_full_pts[:,2],color='green',alpha=0.3, hatch='//',label=r'${\rm 100\%\, Uncertainty}$')

# Rg_ana, nCR_ana=np.loadtxt("nCR_30GeV_ana_YUK04.dat",unpack=True,usecols=[0,1])
# ax.plot(Rg_ana,nCR_ana,'r:',linewidth=3.0)

# Rg_ana, nCR_ana=np.loadtxt("nCR_30GeV_ana_CAB98.dat",unpack=True,usecols=[0,1])
# ax.plot(Rg_ana,nCR_ana,'g:',linewidth=3.0)

# ax.errorbar(Rg_dif,nCR_dif,nCR_dif*0.0,err_Rg_dif-Rg_dif,'o',color='royalblue',markersize=10.0,elinewidth=2.5,label=r'{\rm Diffuse emission}')
# ax.errorbar(Rg_mcl,nCR_mcl,err_nCR_mcl-nCR_mcl,Rg_mcl*0.0,'go',markersize=10.0,elinewidth=2.5,label=r'{\rm Molecular clouds}')
Rg_dif_local=np.array([0.0, 15.0])
nCR_dif_local=np.array([0.453, 0.453])
ax.plot(Rg_dif_local, nCR_dif_local, 'k-', linewidth=3, label=r'{\rm Local CR data}')

ax.set_yscale('log')
ax.legend()
ax.set_xlabel(r'$R_{\odot} {\rm (kpc)}$',fontsize=fs)
ax.set_ylabel(r'$n_{\rm p}(30\,{\rm GeV})\,{ (10^{-13}\, {\rm GeV^{-1}\, cm^{-3}})}$',fontsize=fs)
for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    label_ax.set_fontsize(fs)
ax.set_xlim(0,2)
ax.set_ylim(3.0e-1,3.0e1)
ax.legend(loc='upper left', prop={"size":fs})
ax.grid(linestyle='--')

plt.savefig("fg_CR_profile_sol.png")

################################################################################################
# Plot the CR map 
fig=plt.figure(figsize=(10, 8))
ax=plt.subplot(111)

# c = plt.imshow(np.log10((CR_map_pts.T)*1.0e22), cmap ='magma', extent=[-10, 10, -10, 10], vmin=-1, vmax=1, interpolation ='nearest', origin ='lower') 
c = plt.imshow(np.log10((masked_image.T)*1.0e22), cmap ='magma', extent=[-10, 10, -10, 10], interpolation ='nearest', origin ='lower') 

cbar = plt.colorbar(c)
cbar.set_label(r'${\rm log}_{10}\left[n_{\rm CR}(E)/({\rm 10^{-13}\, GeV^{-1}\, cm^{-3}})\right]$', fontsize=fs)
cbar.ax.tick_params(labelsize=fs)

plt.title(r'$E=30\, {\rm GeV}$', fontsize=fs) 
for label_ax in (ax.get_xticklabels() + ax.get_yticklabels()):
    label_ax.set_fontsize(fs)

plt.savefig("fg_30GeV_sol_mask.png")
