import numpy as np

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


ATilde2=(a0-p0)*sin(2*theta)*cos(phi)+(a1-p1)*sin(phi)*cos(2*theta)+a2*cos(2*theta)
