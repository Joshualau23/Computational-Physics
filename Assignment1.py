# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:16:54 2020

@author: joshu
"""

import math

def quad(x, a, b, c):
    """
    Calculates the output for a quadratic function y=a*x**2+b*x+c
    
    Args:
        x: locations at which the function is to be evaluated
        a: coefficient of x**2
        b: coefficient of x
        c: constant
    
    Returns:
        value of the quadratic function at x 
    """
    
    return(a*x**2 + b*x +c)
    
    
def solve_quad_standard(a,b,c):
    """
    returns the two solutions using the quadratic formula
    -b +/-(b**2-4ac)**(1/2)
    -------
      2 a
      
    Args:
        x: locations at which the function is to be evaluated
        a: coefficient of x**2
        b: coefficient of x
        c: constant
    
    Returns:
        a single tuple of the two solutions (s1,s2)
    """
    ps1 = (-b + math.sqrt(b**2 - 4*a*c) ) / (2.0*a)
    ns2 = (-b - math.sqrt(b**2 - 4*a*c) ) / (2.0*a)
    return (ps1,ns2)


####
    

a=0.001
b=1000.
c=0.001

'''
(s1, s2) = solve_quad_standard(a,b,c)

print(s1, quad(s1, a, b, c))
print(s2, quad(s2, a, b, c))
'''

####

'''
(r1, r2) = solve_quad_standard(-a,-b,-c)
print(r1, quad(r1, -a, -b, -c))
print(r2, quad(r2, -a, -b, -c))
'''
'''
def solve_quad_inv(a,b,c):
    
    """
    returns the two solutions using the quadratic formula
            2c
    ----------------------
    -b -/+(b**2-4ac)**(1/2)
      
    Args:
        x: locations at which the function is to be evaluated
        a: coefficient of x**2
        b: coefficient of x
        c: constant
    
    Returns:
        a single tuple of the two solutions (s1,s2)
    """
    
    inv_ps1 =  (2*c) / ((-b + math.sqrt(b**2 - 4*a*c)))
    inv_ns2 = (2*c) / ((-b - math.sqrt(b**2 - 4*a*c)))
    return (inv_ps1,inv_ns2)

(s1, s2) = solve_quad_inv(a,b,c)
print(s1, quad(s1, a, b, c))
print(s2, quad(s2, a, b, c))
'''

####

def solve_quad(a, b, c):
    
    s1 = (-b-math.sqrt(b**2 - 4*a*c))/(2*a)
    s2 = (-b+math.sqrt(b**2 - 4*a*c))/(2*a)
    return (s1,s2)

(r1, r2) = solve_quad(a,b,c)

print(r1, quad(r1, a, b, c))
print(r2, quad(r2, a, b, c))

(r1, r2) = solve_quad(-a,-b,-c)

print(r1, quad(r1, -a, -b, -c))
print(r2, quad(r2, -a, -b, -c))










