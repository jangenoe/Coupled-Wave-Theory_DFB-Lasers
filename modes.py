import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import optimize
import os

def g(gl,kl):
    return kl*np.sinh(gl)-1j*gl
def g1(gl,kl):
    return kl*np.cosh(gl)-1j
def g2(gl,kl):
    return kl*np.sinh(gl)

def fouriercomp(neff1,neff2,duty,dg):
    n0=neff1+duty*(neff2-neff1)
    dn=(neff2**2-neff1**2)/n0/2.0
    n1=2*dn**2/np.pi*np.sin(np.pi*duty)*dg
    n2=dn/np.pi*np.sin(2*np.pi*duty)
    return n0,np.real(n1),n2

def dispersion2(neff1=1.8,neff2=1.7,gain1=1.1,gain2=0.95):
    n,n1=(neff1+neff2)/2,(neff1-neff2)/2
    a,a1=(gain1+gain2)/2,(gain1-gain2)/2
    kapa=n1+0.5j*a1
    abskapa=np.abs(kapa)
    delta=np.linspace(-2*abskapa,2*abskapa,500)
    gamma1=np.sqrt(kapa**2+(a-1.0j*delta)**2)
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    ax.plot(delta/abskapa,np.imag(gamma1)/abskapa,c='blue')
    ax.plot(delta/abskapa,-np.imag(gamma1)/abskapa,c='blue')
    ax.set_xlabel(r'$\delta / |\kappa |$')
    ax.set_ylabel(r'$Im(\gamma) / | \kappa |$')
    ax.text(0,1.5,r'neff$_1$ ='+str(neff1))
    ax.text(0,1.4,r'neff$_2$ ='+str(neff2))
    ax.text(0,1.3,r'$\alpha_1$ ='+str(gain1))
    ax.text(0,1.2,r'$\alpha_2$ ='+str(gain2))
    ax.set_xlim(-2,2)
    ax.tick_params(direction='in')
    
def modesfinite(kL=-1j,maxmodes=4):
    """
    Aim:
        Calculates the lasing threshold for a set of modes as a function of the dimensionless parameter kL
    parameters:
        * kL: 
        * maxmodes: maximum number of modes to be plotted
    returns:
        the ax of a plot of the modes. The axes can be replotted or rescaled
    """
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    lstl=['-',':']
    SA=['Symmetric','Anti-symmetric']
    if np.real(kL)==0.0:
        kL2=-np.imag(kL)
        aa=[0]+[(0.9+2*m)*1j  for m in range(1,12)]
        kapaLmax=[1]+[abs(np.real((np.pi*a/2)/np.sinh(np.pi*a/2)))*0.997 for a in aa[1:] ]
        index=next(x for x, val in enumerate(kapaLmax)  if val > kL2)
        aa=aa[index:index+maxmodes]
        kapaLmax=kapaLmax[index:index+maxmodes]
        testy=np.linspace(0.01,20,1500)  
        mode=[i for i in range(index,index+maxmodes)]
        gammaL1w=[optimize.newton(g, np.interp(kL2, np.real((-1)**mm*(testy+np.pi*ar/2)/np.sinh(testy+np.pi*ar/2)), testy+np.pi*ar/2), fprime=g1,fprime2=g2, args=(-kL*(-1)**mm, )) for mm,kLm,ar in zip(mode,kapaLmax,aa)]
    else:
        mode=[i for i in range(1,maxmodes+1)]  
        gammaL1w=[optimize.newton(g,mm*(2/np.abs(kL)**0.5+1j*np.pi), fprime=g1,fprime2=g2, args=(-kL*(-1)**mm, )) for mm in mode]                
    for mm,gL,color in zip(mode,gammaL1w,[c for c in mcolors.TABLEAU_COLORS][:maxmodes]):
            xx=(-1)**mm*1j*kL*np.cosh(gL)
            ax.plot([np.imag(-xx)/np.pi,np.imag(-xx)/np.pi],[0,np.real(xx)],c=color,linestyle=lstl[mm%2],label=str(mm)+' '+SA[mm%2]);
            ax.plot([np.imag(xx)/np.pi,np.imag(xx)/np.pi],[0,np.real(xx)],c=color,linestyle=lstl[mm%2]);
    ax.set_xlabel(r'$\delta$ L/ $\pi$')
    ax.set_ylabel(r'$\alpha$ L')
    ax.set_xlim(-maxmodes-np.abs(kL)/np.pi,maxmodes+np.abs(kL)/np.pi)
    ax.set_ylim(0,)
    ax.tick_params(direction='in')
    ax.legend()
    return ax;

def modesfinite2(neff1=1.7,neff2=1.8,duty=0.27, L=8e-6,period=410e-9,dg=0,maxmodes=4,asefile="",dispfile="",minwl=700, maxwl=900, maxangle=20):
    """
    Aim:
        Calculates the dispersion relation and the lasing threshold for a set of modes 
    parameters:
        * neff1: [complex number] effective refractive index of the propagating mode not part of the duty 
        * neff2: [complex number] effective refractive index of the propagating mode part of the duty 
        * duty: fraction [0..1] of period corresponding to neff2
        * L :[m] total length of the active grating
        * period : [m] period of the grating
        * dg : [m] step height corresponding to duty
        * maxmodes: maximum number of modes to be plotted
        * asefile: luminescence spectrum to be plotted on top of the lasing modes {ax[1,0]}
        * dispfile:["*.png"] image file to be plotted on top of the dispersion relation {ax[0,0]}
        * minwl: [nm] only to be used for plotting display file: minimal wavelength on the x-axis of dispfile
        * maxwl: [nm] only to be used for plotting display file: maximal wavelength on the x-axis of dispfile
        * maxangle: [degree] only to be used for plotting display file: maximal angle on the y-axis of dispfile. the angle of the dispersion relation is assumed to be symmetric.
    returns:
        the ax of a plot of the modes. The axes can be replotted or rescaled
        
    notes:
        even modes are symmetric (q=1), antisymmetric modes are not radiating to the surface
    """
    n0,n1,n2=fouriercomp(neff1,neff2,duty,dg)
    n00=np.abs(n0)
    kapa=(n2+1j*n1*np.pi/period)*np.pi/period
    abskapa=np.abs(kapa)
    delta=np.linspace(-2*abskapa-12/L,2*abskapa+12/L,500)
    gamma1=np.sqrt(kapa**2+(2*(np.imag(n0)-n1)/period*np.pi-1.0j*delta)**2)
    kL=kapa*L
    fig, ax = plt.subplots(2,2,figsize=(12,10), constrained_layout=True)
    lstl=['-',':']
    SA=['Symmetric','Anti-symmetric']
    ax[0,0].plot(1e9*n00/(1/period+delta/np.pi/n00),np.imag(gamma1)*period/np.pi*180,c='blue')
    ax[0,0].plot(1e9*n00/(1/period+delta/np.pi/n00),-np.imag(gamma1)*period/np.pi*180,c='blue') 
    ax2 = ax[0,0].twinx()
    ax2.plot(1e9*n00/(1/period+delta/np.pi/n00),np.real(gamma1)*period/np.pi*180,c='red',linestyle=lstl[1])
    ax2.set_ylabel('damping', color='red')  
    ax2.tick_params(direction='in',axis='y', labelcolor='red')    
    if np.real(kL)==0.0:
        kL2=-np.imag(kL)
        aa=[0]+[(0.9+2*m)*1j  for m in range(1,12)]
        kapaLmax=[1]+[abs(np.real((np.pi*a/2)/np.sinh(np.pi*a/2)))*0.997 for a in aa[1:] ]
        index=next(x for x, val in enumerate(kapaLmax)  if val > kL2)
        aa=aa[index:index+maxmodes]
        kapaLmax=kapaLmax[index:index+maxmodes]
        testy=np.linspace(0.01,20,1500)
        mode=[i for i in range(index,index+maxmodes)]
        gammaL1w=[optimize.newton(g, np.interp(kL2, np.real((-1)**mm*(testy+np.pi*ar/2)/np.sinh(testy+np.pi*ar/2)), testy+np.pi*ar/2), fprime=g1,fprime2=g2, args=(-kL*(-1)**mm, )) for mm,kLm,ar in zip(mode,kapaLmax,aa)]
    else:
        mode=[i for i in range(1,maxmodes+1)]
        gammaL1w=[optimize.newton(g,mm*(2/np.abs(kL)**0.5+1j*np.pi), fprime=g1,fprime2=g2, args=(-kL*(-1)**mm, )) for mm in mode]                

    zz=np.linspace(-0.5,0.5,100)
    for mm,gL,color in zip(mode,gammaL1w,[c for c in mcolors.TABLEAU_COLORS][:maxmodes]):
        xx=(-1)**mm*1j*kL*np.cosh(gL)-1j*n1*(np.pi/period)**2
        wlm=1e9*n00/(1/period+np.imag(xx)/np.pi/n00/L)
        wlp=1e9*n00/(1/period-np.imag(xx)/np.pi/n00/L)
        ax[1,0].plot([wlm,wlm],[0,np.real(xx)/L*1e-2],c=color,linestyle=lstl[mm%2],label=str(mm)+' '+SA[mm%2]);
        ax[1,0].plot([wlp,wlp],[0,np.real(xx)/L*1e-2],c=color,linestyle=lstl[mm%2]);
        I=np.abs(np.sinh(gL*(zz+0.5)))**2+np.abs(np.sinh(gL*(zz-0.5)))**2  
        Io=I[0]
        ax[0,1].plot(zz*L*1e6,I/Io,label=str(mm),c=color,linestyle=lstl[mm%2])
        ax[1,1].plot(zz/L/1e6/np.pi*180,np.abs(np.fft.fftshift(np.fft.fft((np.sinh(gL*(zz+0.5))-(-1)**mm*np.sinh(gL*(zz-0.5)))/np.sqrt(Io),n=1500))[700:800])**2,label=str(mm),c=color,linestyle=lstl[mm%2])
    if asefile!="":
        fn=os.getcwd() + "/FAPI-PMAI_120nm_Full_ThinITO_LED_spectra/" + asefile
        ase=np.array([i[5:]  for i in np.genfromtxt(fn,delimiter=',',skip_header=1)]).T
        ax[1,0].twinx().plot(ase[0],ase[1],c='red',linestyle=lstl[1])
    if dispfile!="":
        fn2=os.getcwd() + "/disp_png/" + dispfile
        ax[0,0].imshow(plt.imread(fn2, format='png'), extent=[minwl, maxwl, -maxangle, maxangle])
    ax[0,0].set_xlabel(r'wavelength (nm)')
    ax[0,0].set_ylabel(r'angel (degrees)',color='blue')
    ax[0,0].tick_params(direction='in')
    ax[0,0].tick_params(axis='y', labelcolor='blue')  
    ax[0,0].set_title("dispersion relation prior to lasing")
    ax[0,0].set_xlim(1e9*n00/(1/period+1.2*abskapa/np.pi/n00+2/L),1e9*n00/(1/period-1.2*abskapa/np.pi/n00-2/L))   
    ax[1,0].set_xlabel(r'wavelength (nm)')
    ax[1,0].set_ylabel(r'DC gain $\alpha$ yielding lasing threshold ($cm^{-1}$)')
    ax[1,0].set_title("lasing modes")
    ax[1,0].tick_params(direction='in')
    ax[1,0].set_xlim(1e9*n00/(1/period+1.2*abskapa/np.pi/n00+2/L),1e9*n00/(1/period-1.2*abskapa/np.pi/n00-2/L))
    ax[1,0].set_ylim(0,)
    ax[0,1].set_title("lasing mode intensity")
    ax[0,1].set_xlabel(r'distance along the grating axis ($\mu m$)')
    ax[0,1].set_ylabel(r'intensity')
    ax[0,1].set_xlim(-0.5*L*1e6,0.5*L*1e6)
    ax[0,1].tick_params(direction='in');
    ax[1,1].set_title("lasing far field")
    ax[1,1].set_xlabel(r'angel (degrees)')
    ax[1,1].set_ylabel(r'intensity')
    ax[1,1].set_xlim(-1.0*n00*period/L/np.pi*180,1.0*n00*period/L/np.pi*180)
    ax[1,1].tick_params(direction='in')
    ax[0,1].legend()
    ax[1,1].legend()
    ax[1,0].legend()
    return ax;

def matrix(gL,kL,bL,theta=0,Vlsign=1): # work in progress, see< cite id="kvhuf"><a href="#DOI|10.1002/0470856238">(H. Ghafouri–Shiraz)</a></cite> page 87
    E=np.exp(gL)
    vL=np.emath.sqrt(gL**2-kL**2+1j*0.0)*Vlsign  # sign of real part of sqrt is not controlled in python and jumps!!!
    ro=1j*kL/(vL+gL)
    t11=(E-ro**2/E)*np.exp(-1j*bL)/(1-ro**2)*np.exp(1j*theta)
    t12=-ro*(E-1/E)*np.exp(-1j*bL)/(1-ro**2)*np.exp(-1j*theta)
    t21=ro*(E-1/E)*np.exp(1j*bL)/(1-ro**2)*np.exp(1j*theta)
    t22=-(ro**2*E-1/E)*np.exp(1j*bL)/(1-ro**2)*np.exp(-1j*theta)
    return np.array([[t11,t12],[t21,t22]])
     
def steped_long_array(L=[125e-6,50e-6,50e-6,125e-6]):
    """
    Aim:
        Calculates the dispersion relation and the lasing treshold for a set of modes 
    parameters:
        * L: lengths of the different parts of the structure [non-gain,gain,gain,non-gain]
        * ka: kapa of the gain medium
        * kf: kapa of the non-gain medium
        * b: beta of the period
        * theta: phase jump in the middle of the gain medium
    returns:
        m22 parameter of the matrix, which should get zero to get lasing
    """
    r1=np.matmul(matrix(gL=g*L[2],kL=ka*L[2],bL=b*L[2],theta=0),matrix(gL=g*L[3],kL=kf*L[3],bL=b*L[3],theta=0))
    r2=np.matmul(matrix(gL=gL/2,kL=kL/2,bL=b*L[1],theta=theta),r1)
    r=np.matmul(matrix(gL=gL/2,kL=kL/2,bL=b*L[0],theta=0),r2)
    return r[1,1]
