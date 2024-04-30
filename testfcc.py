import numpy as np
from multiprocessing import Pool
from sympy import *
from stab_bb_correction import stab_BB
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import cKDTree


hf=Rational(1,2)

A0=Matrix([0,0,0])
A1=Matrix([0,hf,hf])
A2=Matrix([hf,0,hf])
A3=Matrix([hf,hf,0])


a0,a1,a2=symbols("a0,a1,a2",cls=Symbol,real=True)
p0,p1=symbols("p0,p1",cls=Symbol,real=True)

theta,phi=symbols("theta,phi",cls=Symbol,real=True)


ATilde0=p0+(a0-p0)*(-hf*cos(2*phi)*cos(2*theta)+hf*cos(2*phi)-hf*cos(2*theta)-hf)\
    +(a1-p1)*(-hf*sin(2*phi)*cos(2*theta)+hf*sin(2*phi))+a2*sin(2*theta)*cos(phi)


ATilde1=p1+(a0-p0)*(-hf*sin(2*phi)*cos(2*theta)+hf*sin(2*phi))\
    +(a1-p1)*(hf*cos(2*phi)*cos(2*theta)-hf*cos(2*phi)-hf*cos(2*theta)-hf)+a2*sin(phi)*sin(2*theta)


ATilde2=(a0-p0)*sin(2*theta)*cos(phi)+(a1-p1)*sin(phi)*sin(2*theta)+a2*cos(2*theta)

##################################functions for point A0
g00=(sin(2*pi*(ATilde0-A0[0])))**2+(cos(2*pi*(ATilde0-A0[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A0[1])))**2+(cos(2*pi*(ATilde1-A0[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A0[2])))**2+(cos(2*pi*(ATilde2-A0[2]))-1)**2

g00=g00.subs([(a0,A0[0]),(a1,A0[1]),(a2,A0[2])])
g01=(sin(2*pi*(ATilde0-A1[0])))**2+(cos(2*pi*(ATilde0-A1[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A1[1])))**2+(cos(2*pi*(ATilde1-A1[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A1[2])))**2+(cos(2*pi*(ATilde2-A1[2]))-1)**2
g01=g01.subs([(a0,A0[0]),(a1,A0[1]),(a2,A0[2])])

g02=(sin(2*pi*(ATilde0-A2[0])))**2+(cos(2*pi*(ATilde0-A2[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A2[1])))**2+(cos(2*pi*(ATilde1-A2[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A2[2])))**2+(cos(2*pi*(ATilde2-A2[2]))-1)**2
g02=g02.subs([(a0,A0[0]),(a1,A0[1]),(a2,A0[2])])

g03=(sin(2*pi*(ATilde0-A3[0])))**2+(cos(2*pi*(ATilde0-A3[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A3[1])))**2+(cos(2*pi*(ATilde1-A3[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A3[2])))**2+(cos(2*pi*(ATilde2-A3[2]))-1)**2
g03=g03.subs([(a0,A0[0]),(a1,A0[1]),(a2,A0[2])])


dg00=[diff(g00,p0),diff(g00,p1),diff(g00,theta),diff(g00,phi)]

dg01=[diff(g01,p0),diff(g01,p1),diff(g01,theta),diff(g01,phi)]

dg02=[diff(g02,p0),diff(g02,p1),diff(g02,theta),diff(g02,phi)]

dg03=[diff(g03,p0),diff(g03,p1),diff(g03,theta),diff(g03,phi)]



g00Np=lambdify((p0,p1,theta,phi),g00,"numpy")

g01Np=lambdify((p0,p1,theta,phi),g01,"numpy")

g02Np=lambdify((p0,p1,theta,phi),g02,"numpy")


g03Np=lambdify((p0,p1,theta,phi),g03,"numpy")


dg00Np=lambdify((p0,p1,theta,phi),dg00,"numpy")

dg01Np=lambdify((p0,p1,theta,phi),dg01,"numpy")

dg02Np=lambdify((p0,p1,theta,phi),dg02,"numpy")

dg03Np=lambdify((p0,p1,theta,phi),dg03,"numpy")

def f0df0(p0,p1,theta,phi):
    g00Val=g00Np(p0,p1,theta,phi)
    g01Val = g01Np(p0, p1, theta, phi)
    g02Val = g02Np(p0, p1, theta, phi)
    g03Val = g03Np(p0, p1, theta, phi)
    gValsTmp=np.array([g00Val,g01Val,g02Val,g03Val])
    inds=np.argsort([g00Val,g01Val,g02Val,g03Val])

    dg00Val=dg00Np(p0,p1,theta,phi)
    dg01Val = dg01Np(p0, p1, theta, phi)
    dg02Val = dg02Np(p0, p1, theta, phi)
    dg03Val = dg03Np(p0, p1, theta, phi)
    # print("ind="+str(inds[0]))
    dgValsTmp=np.array([dg00Val,dg01Val,dg02Val,dg03Val])
    return gValsTmp[inds[0]], dgValsTmp[inds[0],:]

#####################################end of functions for A0



##################################functions for point A1
g10=(sin(2*pi*(ATilde0-A0[0])))**2+(cos(2*pi*(ATilde0-A0[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A0[1])))**2+(cos(2*pi*(ATilde1-A0[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A0[2])))**2+(cos(2*pi*(ATilde2-A0[2]))-1)**2

g10=g10.subs([(a0,A1[0]),(a1,A1[1]),(a2,A1[2])])

g11=(sin(2*pi*(ATilde0-A1[0])))**2+(cos(2*pi*(ATilde0-A1[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A1[1])))**2+(cos(2*pi*(ATilde1-A1[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A1[2])))**2+(cos(2*pi*(ATilde2-A1[2]))-1)**2
g11=g11.subs([(a0,A1[0]),(a1,A1[1]),(a2,A1[2])])

g12=(sin(2*pi*(ATilde0-A2[0])))**2+(cos(2*pi*(ATilde0-A2[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A2[1])))**2+(cos(2*pi*(ATilde1-A2[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A2[2])))**2+(cos(2*pi*(ATilde2-A2[2]))-1)**2
g12=g12.subs([(a0,A1[0]),(a1,A1[1]),(a2,A1[2])])

g13=(sin(2*pi*(ATilde0-A3[0])))**2+(cos(2*pi*(ATilde0-A3[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A3[1])))**2+(cos(2*pi*(ATilde1-A3[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A3[2])))**2+(cos(2*pi*(ATilde2-A3[2]))-1)**2
g13=g13.subs([(a0,A1[0]),(a1,A1[1]),(a2,A1[2])])


dg10=[diff(g10,p0),diff(g10,p1),diff(g10,theta),diff(g10,phi)]

dg11=[diff(g11,p0),diff(g11,p1),diff(g11,theta),diff(g11,phi)]

dg12=[diff(g12,p0),diff(g12,p1),diff(g12,theta),diff(g12,phi)]

dg13=[diff(g13,p0),diff(g13,p1),diff(g13,theta),diff(g13,phi)]



g10Np=lambdify((p0,p1,theta,phi),g10,"numpy")

g11Np=lambdify((p0,p1,theta,phi),g11,"numpy")

g12Np=lambdify((p0,p1,theta,phi),g12,"numpy")


g13Np=lambdify((p0,p1,theta,phi),g13,"numpy")


dg10Np=lambdify((p0,p1,theta,phi),dg10,"numpy")

dg11Np=lambdify((p0,p1,theta,phi),dg11,"numpy")

dg12Np=lambdify((p0,p1,theta,phi),dg12,"numpy")

dg13Np=lambdify((p0,p1,theta,phi),dg13,"numpy")

def f1df1(p0,p1,theta,phi):
    g10Val=g10Np(p0,p1,theta,phi)
    g11Val = g11Np(p0, p1, theta, phi)
    g12Val = g12Np(p0, p1, theta, phi)
    g13Val = g13Np(p0, p1, theta, phi)
    gValsTmp=np.array([g10Val,g11Val,g12Val,g13Val])
    inds=np.argsort(gValsTmp)

    dg10Val=dg10Np(p0,p1,theta,phi)
    dg11Val = dg11Np(p0, p1, theta, phi)
    dg12Val = dg12Np(p0, p1, theta, phi)
    dg13Val = dg13Np(p0, p1, theta, phi)

    dgValsTmp=np.array([dg10Val,dg11Val,dg12Val,dg13Val])
    return gValsTmp[inds[0]], dgValsTmp[inds[0],:]

#####################################end of functions for A1

##################################functions for point A2
g20=(sin(2*pi*(ATilde0-A0[0])))**2+(cos(2*pi*(ATilde0-A0[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A0[1])))**2+(cos(2*pi*(ATilde1-A0[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A0[2])))**2+(cos(2*pi*(ATilde2-A0[2]))-1)**2

g20=g20.subs([(a0,A2[0]),(a1,A2[1]),(a2,A2[2])])

g21=(sin(2*pi*(ATilde0-A1[0])))**2+(cos(2*pi*(ATilde0-A1[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A1[1])))**2+(cos(2*pi*(ATilde1-A1[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A1[2])))**2+(cos(2*pi*(ATilde2-A1[2]))-1)**2
g21=g21.subs([(a0,A2[0]),(a1,A2[1]),(a2,A2[2])])

g22=(sin(2*pi*(ATilde0-A2[0])))**2+(cos(2*pi*(ATilde0-A2[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A2[1])))**2+(cos(2*pi*(ATilde1-A2[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A2[2])))**2+(cos(2*pi*(ATilde2-A2[2]))-1)**2
g22=g22.subs([(a0,A2[0]),(a1,A2[1]),(a2,A2[2])])

g23=(sin(2*pi*(ATilde0-A3[0])))**2+(cos(2*pi*(ATilde0-A3[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A3[1])))**2+(cos(2*pi*(ATilde1-A3[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A3[2])))**2+(cos(2*pi*(ATilde2-A3[2]))-1)**2
g23=g23.subs([(a0,A2[0]),(a1,A2[1]),(a2,A2[2])])


dg20=[diff(g20,p0),diff(g20,p1),diff(g20,theta),diff(g20,phi)]

dg21=[diff(g21,p0),diff(g21,p1),diff(g21,theta),diff(g21,phi)]

dg22=[diff(g22,p0),diff(g22,p1),diff(g22,theta),diff(g22,phi)]

dg23=[diff(g23,p0),diff(g23,p1),diff(g23,theta),diff(g23,phi)]



g20Np=lambdify((p0,p1,theta,phi),g20,"numpy")

g21Np=lambdify((p0,p1,theta,phi),g21,"numpy")

g22Np=lambdify((p0,p1,theta,phi),g22,"numpy")


g23Np=lambdify((p0,p1,theta,phi),g23,"numpy")


dg20Np=lambdify((p0,p1,theta,phi),dg20,"numpy")

dg21Np=lambdify((p0,p1,theta,phi),dg21,"numpy")

dg22Np=lambdify((p0,p1,theta,phi),dg22,"numpy")

dg23Np=lambdify((p0,p1,theta,phi),dg23,"numpy")

def f2df2(p0,p1,theta,phi):
    g20Val=g20Np(p0,p1,theta,phi)
    g21Val = g21Np(p0, p1, theta, phi)
    g22Val = g22Np(p0, p1, theta, phi)
    g23Val = g23Np(p0, p1, theta, phi)
    gValsTmp=np.array([g20Val,g21Val,g22Val,g23Val])
    inds=np.argsort(gValsTmp)

    dg20Val=dg20Np(p0,p1,theta,phi)
    dg21Val = dg21Np(p0, p1, theta, phi)
    dg22Val = dg22Np(p0, p1, theta, phi)
    dg23Val = dg23Np(p0, p1, theta, phi)

    dgValsTmp=np.array([dg20Val,dg21Val,dg22Val,dg23Val])
    return gValsTmp[inds[0]], dgValsTmp[inds[0],:]

#####################################end of functions for A2

##################################functions for point A3
g30=(sin(2*pi*(ATilde0-A0[0])))**2+(cos(2*pi*(ATilde0-A0[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A0[1])))**2+(cos(2*pi*(ATilde1-A0[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A0[2])))**2+(cos(2*pi*(ATilde2-A0[2]))-1)**2

g30=g30.subs([(a0,A3[0]),(a1,A3[1]),(a2,A3[2])])

g31=(sin(2*pi*(ATilde0-A1[0])))**2+(cos(2*pi*(ATilde0-A1[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A1[1])))**2+(cos(2*pi*(ATilde1-A1[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A1[2])))**2+(cos(2*pi*(ATilde2-A1[2]))-1)**2
g31=g31.subs([(a0,A3[0]),(a1,A3[1]),(a2,A3[2])])

g32=(sin(2*pi*(ATilde0-A2[0])))**2+(cos(2*pi*(ATilde0-A2[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A2[1])))**2+(cos(2*pi*(ATilde1-A2[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A2[2])))**2+(cos(2*pi*(ATilde2-A2[2]))-1)**2
g32=g32.subs([(a0,A3[0]),(a1,A3[1]),(a2,A3[2])])

g33=(sin(2*pi*(ATilde0-A3[0])))**2+(cos(2*pi*(ATilde0-A3[0]))-1)**2\
    +(sin(2*pi*(ATilde1-A3[1])))**2+(cos(2*pi*(ATilde1-A3[1]))-1)**2\
    +(sin(2*pi*(ATilde2-A3[2])))**2+(cos(2*pi*(ATilde2-A3[2]))-1)**2
g33=g33.subs([(a0,A3[0]),(a1,A3[1]),(a2,A3[2])])


dg30=[diff(g30,p0),diff(g30,p1),diff(g30,theta),diff(g30,phi)]

dg31=[diff(g31,p0),diff(g31,p1),diff(g31,theta),diff(g31,phi)]

dg32=[diff(g32,p0),diff(g32,p1),diff(g32,theta),diff(g32,phi)]

dg33=[diff(g33,p0),diff(g33,p1),diff(g33,theta),diff(g33,phi)]



g30Np=lambdify((p0,p1,theta,phi),g30,"numpy")

g31Np=lambdify((p0,p1,theta,phi),g31,"numpy")

g32Np=lambdify((p0,p1,theta,phi),g32,"numpy")


g33Np=lambdify((p0,p1,theta,phi),g33,"numpy")


dg30Np=lambdify((p0,p1,theta,phi),dg30,"numpy")

dg31Np=lambdify((p0,p1,theta,phi),dg31,"numpy")

dg32Np=lambdify((p0,p1,theta,phi),dg32,"numpy")

dg33Np=lambdify((p0,p1,theta,phi),dg33,"numpy")

def f3df3(p0,p1,theta,phi):
    g30Val=g30Np(p0,p1,theta,phi)
    g31Val = g31Np(p0, p1, theta, phi)
    g32Val = g32Np(p0, p1, theta, phi)
    g33Val = g33Np(p0, p1, theta, phi)
    gValsTmp=np.array([g30Val,g31Val,g32Val,g33Val])
    inds=np.argsort(gValsTmp)

    dg30Val=dg30Np(p0,p1,theta,phi)
    dg31Val = dg31Np(p0, p1, theta, phi)
    dg32Val = dg32Np(p0, p1, theta, phi)
    dg33Val = dg33Np(p0, p1, theta, phi)

    dgValsTmp=np.array([dg30Val,dg31Val,dg32Val,dg33Val])
    return gValsTmp[inds[0]], dgValsTmp[inds[0],:]
#####################################end of functions for A3


def cdc(lmd):
    p0,p1,theta,phi=lmd
    f0,df0=f0df0(p0,p1,theta,phi)
    f1,df1=f1df1(p0,p1,theta,phi)
    f2,df2=f2df2(p0,p1,theta,phi)
    f3,df3=f3df3(p0,p1,theta,phi)
    funcVal=f0+f1+f2+f3
    direction=df0+df1+df2+df3

    return funcVal,direction


p0ValsAll=np.linspace(0.1,1,5)
p1ValsAll=np.linspace(0.1,1,5)
thetaValsAll=np.linspace(0.1,np.pi,5)
phiValsAll=np.linspace(0.1,2*np.pi,5)

inInitVals=[[p0val,p1val,thetaval,phival] for p0val in p0ValsAll for p1val in p1ValsAll for thetaval in thetaValsAll for phival in phiValsAll]

def mod1(val):
    epsRel = 1e-6
    epsAbs = 1e-7
    # if the axis lies on x-y plane
    val=val%1
    if np.isclose(val, 1, rtol=epsRel, atol=epsAbs, equal_nan=False):
        val=0
    if np.isclose(val, 0, rtol=epsRel, atol=epsAbs, equal_nan=False):
        val=0
    return val

def modpi(val):
    epsRel = 1e-6
    epsAbs = 1e-7
    # if the axis lies on x-y plane
    val = val % np.pi
    if np.isclose(val, np.pi, rtol=epsRel, atol=epsAbs, equal_nan=False):
        val = 0
    if np.isclose(val, 0, rtol=epsRel, atol=epsAbs, equal_nan=False):
        val = 0
    return val
def obj_func(lmd):
    funcVal,_=cdc(lmd)
    return funcVal

def obj_grad(lmd):
    _,grad=cdc(lmd)
    return grad

def search_min(lmd):
    # tBBStart = datetime.now()
    x, val = stab_BB(lmd, obj_func, obj_grad, c=0.1)
    # tBBEnd = datetime.now()
    # print("bb time: ", tBBEnd - tBBStart)
    return [x, val]

def result2StandardForm(lmd):
    """

    :param lmd: [p0,p1,theta,phi] from BB line search
    :return:
    """
    p0,p1,theta,phi=lmd
    epsRel=1e-6
    epsAbs=1e-6
    #if the axis lies on x-y plane
    if np.isclose(np.cos(theta), 0, rtol=epsRel, atol=epsAbs, equal_nan=False):
        #if the rotation axis is parallel to x axis
        if np.isclose(np.sin(phi),0,rtol=epsRel, atol=epsAbs, equal_nan=False):
            #we take the intersection point of the rotation axis with y axis
            lmdStandard=[0,mod1(p1),modpi(theta),modpi(phi)]
            return lmdStandard
        #if the rotation axis is parallel to y axis
        if np.isclose(np.cos(phi),0,rtol=epsRel, atol=epsAbs, equal_nan=False):
            #we take the intersection point of the rotation axis with x axis
            lmdStandard = [mod1(p0),0,modpi(theta),modpi(phi)]
            return lmdStandard
        #if the rotation axis intersects both x and y axis, we we take the intersection point of the rotation axis with x axis
        xVal=p0-p1*np.cos(phi)/np.sin(phi)
        lmdStandard =[mod1(xVal),0,modpi(theta),modpi(phi)]
        return lmdStandard

    # if the axis does not lie on x-y plane
    #if the axis is orthogonal to x-y plane
    if np.isclose(np.sin(theta), 0, rtol=epsRel, atol=epsAbs, equal_nan=False):
        lmdStandard =[mod1(p0),mod1(p1),0,0]
        return lmdStandard
    lmdStandard=[mod1(p0),mod1(p1),modpi(theta),modpi(phi)]
    return lmdStandard



vecsAll=[]

# counter=0
# for lmdTmp in inInitVals:
#     x,val=search_min(lmdTmp)
#
#     if np.abs(obj_func(x))<1e-10:
#         print("computation " + str(counter))
#         xStandard=result2StandardForm(x)
#         vecsAll.append(xStandard)
#     counter+=1
#################################parallel execution
tParallelStart=datetime.now()
procNum=48
pool0=Pool(procNum)
retAll=pool0.map(search_min,inInitVals)
resultsAll=[]
for item in retAll:
    x,_=item
    if np.abs(obj_func(x)) < 1e-10:
        xStandard = result2StandardForm(x)
        vecsAll.append(xStandard)
tParallelEnd=datetime.now()
print("parallel time: ",tParallelEnd-tParallelStart)
###################################
vecsAll=np.array(vecsAll)

tree = cKDTree(vecsAll)
threshHold=1e-7
# Find pairs with distances less than the threshold
pairs = tree.query_pairs(r=threshHold, output_type='ndarray')

# Create a mask to filter out duplicates
keep = np.ones(len(vecsAll), dtype=bool)
for i, j in pairs:
    keep[j] = False  # Mark the second item in each pair as duplicate

# Filter the vectors
filtered_vectors = vecsAll[keep]
for vec in filtered_vectors:
    print(vec)
    print("#############")
################################
# lmdTmp=inInitVals[10]
# x,val=search_min(lmdTmp)
# print(x)
# print(obj_func(x))

################################

#plot cost function against p0


# p1Val=0.5
# thetaVal=0.0*np.pi
# phiVal=0.98*np.pi
#
# p0ValsAll=np.linspace(0,1,50)
# lmdAll=[[p0Val,p1Val,thetaVal,phiVal] for p0Val in p0ValsAll]
#
# cdcValsAll=[cdc(lmd) for lmd in lmdAll]
#
# pltcValsAll=[item[0] for item in cdcValsAll]
#
# plt.figure()
# plt.plot(p0ValsAll,pltcValsAll,color="black")
# plt.xlabel("$p_{0}$")
# plt.savefig("p0tmp.png")
# plt.close()

############################end of plotting p0


#plot cost function against p1
# p1ValsAll=np.linspace(0,1,50)
#
# p0Val=0.23
# thetaVal=0.44*np.pi
# phiVal=0.25*np.pi
# lmdAll=[[p0Val,p1Val,thetaVal,phiVal] for p1Val in p1ValsAll]
# cdcValsAll=[cdc(lmd) for lmd in lmdAll]
#
# pltcValsAll=[item[0] for item in cdcValsAll]
#
# plt.figure()
# plt.plot(p1ValsAll,pltcValsAll,color="black")
# plt.xlabel("$p_{1}$")
# plt.savefig("p1tmp.png")
# plt.close()

############################end of plotting p1


#plot cost function against theta

# thetaValsAll=np.linspace(0,np.pi,50)
#
# p0Val=0.77
# p1Val=0.51
# phiVal=0.68*np.pi
# lmdAll=[[p0Val,p1Val,thetaVal,phiVal] for thetaVal in thetaValsAll]
#
# cdcValsAll=[cdc(lmd) for lmd in lmdAll]
#
# pltcValsAll=[item[0] for item in cdcValsAll]
#
# plt.figure()
# plt.plot(thetaValsAll/np.pi,pltcValsAll,color="black")
# plt.xlabel("$\\theta/\pi$")
# plt.savefig("thetatmp.png")
# plt.close()

############################end of plotting theta

#plot cost function against phi
# phiValsAll=np.linspace(0,2*np.pi,100)
#
# p0Val=0.74
# p1Val=0.41
# thetaVal=0.82*np.pi
# lmdAll=[[p0Val,p1Val,thetaVal,phiVal] for phiVal in phiValsAll]
#
# cdcValsAll=[cdc(lmd) for lmd in lmdAll]
#
# pltcValsAll=[item[0] for item in cdcValsAll]
#
# plt.figure()
# plt.plot(phiValsAll/np.pi,pltcValsAll,color="black")
# plt.xlabel("$\phi/\pi$")
# plt.savefig("phitmp.png")
# plt.close()

############################end of plotting phi