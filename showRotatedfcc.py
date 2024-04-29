import numpy as np
from multiprocessing import Pool
from stab_bb_correction import stab_BB
import warnings
import matplotlib.pyplot as plt

n=2
#this script computes coordinate of rotated point

def mod1(val):
    if np.abs(val-1)<1e-8:
        return 0
    else:
        return val

def rotatedPoint(p0,p1,p2,theta,phi,A):
    """


    :param A: point to be rotated
    :return: rotated point
    """
    # p0,p1,theta,phi=lmd
    F=np.array([
        [np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi),-np.sin(phi)],
        [np.sin(theta)*np.sin(phi),np.cos(theta)*np.sin(phi),np.cos(phi)],
        [np.cos(theta),-np.sin(theta),0]
    ])
    FInv=F.transpose()
    Rn=np.array([
        [1,0,0],
        [0,np.cos(2*np.pi/n),-np.sin(2*np.pi/n)],
        [0,np.sin(2*np.pi/n),np.cos(2*np.pi/n)]
    ])
    rn=F@Rn@FInv

    P=np.array([p0,p1,p2])

    ATilde=P+rn@(A-P)

    ATilde=ATilde%1

    AMod=np.array([mod1(ATilde[j]) for j in range(0,len(ATilde))])
    return AMod



lmdTmp=[ 1.00821650e-01, -3.39022356e-15,  1.57079633e+00,  6.28318531e+00]
p0,p1,theta,phi=lmdTmp



hf=1/2

A0=np.array([0,0,0])
A1=np.array([0,hf,hf])
A2=np.array([hf,0,hf])
A3=np.array([hf,hf,0])

p2=0
print("p0="+str(p0))
print("theta="+str((theta%np.pi)/np.pi)+"pi")
print("phi="+str((phi%np.pi)/np.pi)+"pi")
# print("angle="+str(phi/np.pi)+"pi")
a=np.array([1,0,0])
b=np.array([0,1,0])
c=np.array([0,0,1])
print(rotatedPoint(p0,p1,p2,theta,phi,A0))