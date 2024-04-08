from sympy import *
from sympy.simplify.fu import TR8


theta,phi=symbols('theta,phi',cls=Symbol)


F=Matrix([[sin(theta)*cos(phi),cos(theta)*cos(phi),-sin(phi)],
          [sin(theta)*sin(phi),cos(theta)*sin(phi),cos(phi)],
          [cos(theta),-sin(theta),0]])

FInv=F.transpose()

n=2#symbols("n",cls=Symbol)

Rn=Matrix([[1,0,0],
           [0,cos(2*pi/n),-sin(2*pi/n)],
           [0,sin(2*pi/n),cos(2*pi/n)]])


r=F*Rn*FInv

pprint(r[2,2])