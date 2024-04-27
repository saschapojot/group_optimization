import numpy as np
from multiprocessing import Pool
from sympy import *


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
    +(a1-p1)*(hf*cos(2*phi)*cos(2*theta)-hf*cos(2*phi)-hf*cos(2*theta)-hf)+a2*sin(phi)*cos(2*theta)


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
    gValsTmp=[g00Val,g01Val,g02Val,g03Val]
    inds=np.argsort([g00Val,g01Val,g02Val,g03Val])

    dg00Val=dg00Np(p0,p1,theta,phi)
    dg01Val = dg01Np(p0, p1, theta, phi)
    dg02Val = dg02Np(p0, p1, theta, phi)
    dg03Val = dg03Np(p0, p1, theta, phi)

    dgValsTmp=[dg00Val,dg01Val,dg02Val,dg03Val]
    return gValsTmp[inds[0]], dgValsTmp[inds[0]]

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
    gValsTmp=[g10Val,g11Val,g12Val,g13Val]
    inds=np.argsort(gValsTmp)

    dg10Val=dg10Np(p0,p1,theta,phi)
    dg11Val = dg11Np(p0, p1, theta, phi)
    dg12Val = dg12Np(p0, p1, theta, phi)
    dg13Val = dg13Np(p0, p1, theta, phi)

    dgValsTmp=[dg10Val,dg11Val,dg12Val,dg13Val]
    return gValsTmp[inds[0]], dgValsTmp[inds[0]]

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
    gValsTmp=[g20Val,g21Val,g22Val,g23Val]
    inds=np.argsort(gValsTmp)

    dg20Val=dg20Np(p0,p1,theta,phi)
    dg21Val = dg21Np(p0, p1, theta, phi)
    dg22Val = dg22Np(p0, p1, theta, phi)
    dg23Val = dg23Np(p0, p1, theta, phi)

    dgValsTmp=[dg20Val,dg21Val,dg22Val,dg23Val]
    return gValsTmp[inds[0]], dgValsTmp[inds[0]]

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
    gValsTmp=[g30Val,g31Val,g32Val,g33Val]
    inds=np.argsort(gValsTmp)

    dg30Val=dg30Np(p0,p1,theta,phi)
    dg31Val = dg31Np(p0, p1, theta, phi)
    dg32Val = dg32Np(p0, p1, theta, phi)
    dg33Val = dg33Np(p0, p1, theta, phi)

    dgValsTmp=[dg30Val,dg31Val,dg32Val,dg33Val]
    return gValsTmp[inds[0]], dgValsTmp[inds[0]]
#####################################end of functions for A3


def cdc(lmd):
    p0,p1,theta,phi=lmd
    f0,df0=f0df0(p0,p1,theta,phi)
    f1,df1=f1df1(p0,p1,theta,phi)
    f2,df2=f2df2(p0,p1,theta,phi)
    f3,df3=f3df3(p0,p1,theta,phi)
    funcVal=f0+f1+f2+f3
    direction=[df0,df1,df2,df3]

    return funcVal,direction


p0ValsAll=np.linspace(0.1,1,10)
p1ValsAll=np.linspace(0.1,1,10)
thetaValsAll=np.linspace(0.1,np.pi,10)
phiValsAll=np.linspace(0.1,2*np.pi,10)

inInitVals=[[p0val,p1val,thetaval,phival] for p0val in p0ValsAll for p1val in p1ValsAll for thetaval in thetaValsAll for phival in phiValsAll]


procNum=48

pool0=Pool(procNum)

