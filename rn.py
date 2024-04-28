from sympy import *
from sympy.simplify.fu import TR8, TR7

#this script computes rotation matrix rn


theta,phi=symbols("theta,phi",cls=Symbol,real=True)


F=Matrix([[sin(theta)*cos(phi),cos(theta)*cos(phi),-sin(phi)],
          [sin(theta)*sin(phi),cos(theta)*sin(phi),cos(phi)],
          [cos(theta),-sin(theta),0]])

FInv=F.transpose()

n=2

Rn=Matrix([[1,0,0],
           [0,cos(2*pi/n),-sin(2*pi/n)],
           [0,sin(2*pi/n),cos(2*pi/n)]])


rn=F*Rn*FInv
hf=Rational(1,2)

x=Matrix([[-hf*cos(2*phi)*cos(2*theta)+hf*cos(2*phi)-hf*cos(2*theta)-hf,hf*sin(2*phi)-hf*sin(2*phi)*cos(2*theta),sin(2*theta)*cos(phi)],
          [hf*sin(2*phi)-hf*sin(2*phi)*cos(2*theta),hf*cos(2*phi)*cos(2*theta)-hf*cos(2*phi)-hf*cos(2*theta)-hf,sin(phi)*sin(2*theta)],
          [cos(phi)*sin(2*theta),sin(phi)*sin(2*theta),cos(2*theta)]])


