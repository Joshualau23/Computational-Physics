# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:08:35 2020

@author: joshu
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.signal import argrelmin, argrelmax

from math import pi,sqrt,sin,cos


class Pendulum(object):
    """
    This class the motion of a DDP Pendulum.
    
    Data Attributes:
        omega_0
        q 
        A
        omega_d
        period_0
        period_d
        resonance_amp
    
    Methods:
        dpdt - returns the derivative of w, where w is the derivative of theta with respect to time
        ode_integrator - ODE solver using odeint function from scipy.integrate
    
    """

    
    def __init__(self,omega_0=1., q=0., A=0., omega_d=0.):
        """
        DDP Pendulum
        
        Parameters:
        -----------
        omega_0
        q 
        A
        omega_d
        """
        
        #self.p0 = p0
        #self.t = t
        self.omega_0 = omega_0         
        self.q = q               
        self.A = A
        self.omega_d = omega_d
        self.period_0 = 2*pi / (omega_0)
        if omega_d == 0:     
            self.period_d = 0
        else:
            self.period_d = 2*pi / (omega_d)
        self.resonance_amp = A / sqrt( (omega_0**2 - omega_d**2)**2 + q**2 * omega_d**2)
        
    
    def dpdt(self,p0,t=0):
        intial_cond = -(self.omega_0**2 * sin(p0[0])) - self.q*p0[1] + self.A*cos(self.omega_d*t)
        dpdt_func = np.array([p0[1], intial_cond])
        return dpdt_func
    
    
    def ode_integrator(self,p0,t):
        values = odeint(self.dpdt, p0, t)
        return values
    


#########################################
#c



#p = [[Initial Angle, Initial Angular Velocity]]
initial_angle = 1 * (pi / 180.0)
p = np.array([initial_angle,0])

period = 2*pi / (1.0)
endtime = period*6
steps = 10000
time = np.linspace(0,endtime, steps)
#Pendulum(omega, q, A, omega_d)
DDP_Pen = Pendulum(1.0,0.,0.,0.)
solution = DDP_Pen.ode_integrator(p,time)

max_list = argrelmax(solution[:,0]) 
min_list = argrelmin(solution[:,0])

for x_coord in max_list[0]:
    plt.axvline(x=x_coord*endtime / steps,color = "black", linestyle = "dashed")
for x_coord in min_list[0]:
    plt.axvline(x=x_coord*endtime / steps,color = "black", linestyle = "dashed")
    
    
plt.plot(time,solution[:,0], color = "Gold")
plt.legend(loc= "upper left")
plt.title("Simple Harmonic Pendulum" )
plt.xlabel("Time (sec)")
plt.ylabel("Angle (rad)")
plt.grid()   
plt.show()
    
    
'''

#########################################
#d


#p = [[Initial Angle, Initial Angular Velocity]]
initial_angle_1 = 1.0 * (pi / 180.0)
initial_angle_2 = 15.0 * (pi / 180.0)
initial_angle_3 = 90.0 * (pi / 180.0)
initial_angle_4 = 179.0 * (pi / 180.0)


p_1 = np.array([initial_angle_1,0])
p_2 = np.array([initial_angle_2,0])
p_3 = np.array([initial_angle_3,0])
p_4 = np.array([initial_angle_4,0])


endtime = 8*2*pi
steps = 10000
time = np.linspace(0,endtime, steps)
#Pendulum(omega, q, A, omega_d)

#Calling the general Pendulum Class
DDP_Pen = Pendulum(1.0,0.,0.,0.)

#Creating Seperate list for each Pendulum
solution_1 = DDP_Pen.ode_integrator(p_1,time)
solution_2 = DDP_Pen.ode_integrator(p_2,time)
solution_3 = DDP_Pen.ode_integrator(p_3,time)
solution_4 = DDP_Pen.ode_integrator(p_4,time)


#Finding the max and min of 4 pendulum
max_list_1 = argrelmax(solution_1[:,0]) 
min_list_1 = argrelmin(solution_1[:,0])
for x_coord in max_list_1[0]:
    plt.axvline(x=x_coord*endtime / steps,color = "Gold", linestyle = "dashed")
for x_coord in min_list_1[0]:
    plt.axvline(x=x_coord*endtime / steps,color = "Gold", linestyle = "dashed")

max_list_2 = argrelmax(solution_2[:,0]) 
min_list_2 = argrelmin(solution_2[:,0])
for x_coord in max_list_2[0]:
    plt.axvline(x=x_coord*endtime / steps,color = "mediumblue", linestyle = "dashed")
for x_coord in min_list_2[0]:
    plt.axvline(x=x_coord*endtime / steps,color = "mediumblue", linestyle = "dashed")


max_list_3 = argrelmax(solution_3[:,0]) 
min_list_3 = argrelmin(solution_3[:,0])
for x_coord in max_list_3[0]:
    plt.axvline(x=x_coord*endtime / steps,color = "limegreen", linestyle = "dashed")
for x_coord in min_list_3[0]:
    plt.axvline(x=x_coord*endtime / steps,color = "limegreen", linestyle = "dashed")


max_list_4 = argrelmax(solution_4[:,0]) 
min_list_4 = argrelmin(solution_4[:,0])
for x_coord in max_list_4[0]:
    plt.axvline(x=x_coord*endtime / steps,color = "orangered", linestyle = "dashed")
for x_coord in min_list_4[0]:
    plt.axvline(x=x_coord*endtime / steps,color = "orangered", linestyle = "dashed")


plt.plot(time,solution_1[:,0] / initial_angle_1, color = "Gold", Label = " $ \Theta _0 = 1 $")
#plt.plot(time,solution_2[:,0] / initial_angle_2, color = "mediumblue", Label = " $ \Theta _0 = 15 $")
plt.plot(time,solution_3[:,0] / initial_angle_3, color = "limegreen", Label = " $ \Theta _0 = 90 $")
plt.plot(time,solution_4[:,0] / initial_angle_4, color = "orangered", Label = " $ \Theta _0 = 179 $")
plt.legend(loc= "upper left")
plt.title("Anharmonic Oscillators" )
plt.xlabel("Time (sec)")
plt.ylabel("Angle (rad) $\Theta _0$")
plt.grid()  
plt.show() 

'''
    
#########################################
#e

'''
#p = [[Initial Angle, Initial Angular Velocity]]
initial_angle = 1.0 * (pi / 100.0)



p = np.array([initial_angle,0])


endtime = 4*2*pi
steps = 10000
time = np.linspace(0,endtime, steps)

#Pendulum(omega, q, A, omega_d)
#Calling the general Pendulum Class
DDP_Pen_1 = Pendulum(1.0,0.1,0.,0.)
DDP_Pen_2 = Pendulum(1.0,0.2,0.,0.)
DDP_Pen_3 = Pendulum(1.0,0.5,0.,0.)
DDP_Pen_4 = Pendulum(1.0,2.,0.,0.)
DDP_Pen_5 = Pendulum(1.0,3.,0.,0.)

#Creating Seperate list for each Pendulum
solution_1 = DDP_Pen_1.ode_integrator(p,time)
solution_2 = DDP_Pen_2.ode_integrator(p,time)
solution_3 = DDP_Pen_3.ode_integrator(p,time)
solution_4 = DDP_Pen_4.ode_integrator(p,time)
solution_5 = DDP_Pen_5.ode_integrator(p,time)



#plt.plot(time,solution_1[:,0], color = "Gold", Label = "q = 0.1")
#plt.plot(time,solution_2[:,0], color = "mediumblue", Label = "q = 0.2")
#plt.plot(time,solution_3[:,0], color = "limegreen", Label = "q = 0.5")
plt.plot(time,solution_4[:,0], color = "orangered", Label = "q = 2.0")
plt.plot(time,solution_5[:,0], color = "darkviolet", Label = "q = 3.0")
plt.legend(loc= "upper left")
plt.title("Damped Oscillators" )
plt.xlabel("Time (sec)")
plt.ylabel("Angle (rad)")
plt.grid()  
plt.show()  


































#########################################
#f  



#p = [[Initial Angle, Initial Angular Velocity]]
initial_angle_1 = 0 * (pi / 180.0)
initial_angle_2 = 30.0 * (pi / 180.0)
initial_angle_3 = 90.0 * (pi / 180.0)


p_1 = np.array([initial_angle_1,0])
p_2 = np.array([initial_angle_2,0])
p_3 = np.array([initial_angle_3,0])


period = 2*pi / (2./3.)
endtime = period*8
steps = 10000
time = np.linspace(0,endtime, steps)
#Pendulum(omega, q, A, omega_d)

#Calling the general Pendulum Class
DDP_Pen = Pendulum(1.0,0.5,0.5,2./3.)

#Creating Seperate list for each Pendulum
solution_1 = DDP_Pen.ode_integrator(p_1,time)
solution_2 = DDP_Pen.ode_integrator(p_2,time)
solution_3 = DDP_Pen.ode_integrator(p_3,time)

#Drawing period of the Driving Force

driving_period = np.arange(0,endtime,period)
for x_coord in driving_period:
    plt.axvline(x=x_coord ,color = "black", linestyle = "dashed")





plt.plot(time,solution_1[:,0] , color = "Gold", Label = " $ \Theta _0 = 0 $")
plt.plot(time,solution_2[:,0] , color = "mediumblue", Label = " $ \Theta _0 = 30 $")
plt.plot(time,solution_3[:,0] , color = "limegreen", Label = " $ \Theta _0 = 90 $")
plt.legend(loc= "upper left")
plt.title("Anharmonic Oscillators" )
plt.xlabel("Time (sec)")
plt.ylabel("Angle (rad) ")
plt.grid()  
plt.show() 

'''
    
#########################################
#g part 1 

'''    

#p = [[Initial Angle, Initial Angular Velocity]]
initial_angle = 0 * (pi / 180.0)



p = np.array([initial_angle,0])

period_1 = 2*pi / (0.9)
period_2 = 2*pi / (1.0)
period_3 = 2*pi / (1.1)


endtime_1 = period_1*10
endtime_2 = period_2*10
endtime_3 = period_3*10


steps = 10000
time_1 = np.linspace(0,endtime_1, steps)
time_2 = np.linspace(0,endtime_2, steps)
time_3 = np.linspace(0,endtime_3, steps)


#Pendulum(omega, q, A, omega_d)
#Calling the general Pendulum Class
DDP_Pen_1 = Pendulum(1.0,0.2,0.1,0.9)
DDP_Pen_2 = Pendulum(1.0,0.2,0.1,1.0)
DDP_Pen_3 = Pendulum(1.0,0.2,0.1,1.1)


#Creating Seperate list for each Pendulum
solution_1 = DDP_Pen_1.ode_integrator(p,time_1)
solution_2 = DDP_Pen_2.ode_integrator(p,time_2)
solution_3 = DDP_Pen_3.ode_integrator(p,time_3)


plt.figure()
plt.plot(time_1,solution_1[:,0] , color = "Gold", Label = "$ \omega _d = 0.9 $")
plt.plot(time_2,solution_2[:,0] , color = "mediumblue", Label = " $ \omega _d = 1.0 $")
plt.plot(time_3,solution_3[:,0] , color = "limegreen", Label = " $ \omega _d = 1.1 $")
plt.legend(loc= "upper left")
plt.title("Resonance" )
plt.xlabel("Time (sec)")
plt.ylabel("Angle (rad)")
plt.grid()  
plt.show()
    


#########################################
#g part 2


#For q = 0.2
amplitude_list_1 = []
theta_res_list_1 = []

#For q = 0.4
amplitude_list_2 = []
theta_res_list_2 = []

#For q = 0.5
amplitude_list_3 = []
theta_res_list_3 = []

omega_driving = np.arange(0.5,1.52,0.02)

initial_angle = 0 * (pi / 180.0)
p = np.array([initial_angle,0])



for i in omega_driving:
    
    time = np.linspace(0, 20*(2*pi / i), 20000)   
    
    ###For q = 0.2
    
    #Creates time list depending on what the value of omega_d.
    DDP_Pen = Pendulum(1.0,0.2,0.1,i)
    #Grabs the theoretical theta_res value within the Pendulum class
    theoretical_theta = DDP_Pen.resonance_amp
    solution = DDP_Pen.ode_integrator(p,time)
    
    #Get the min and max of the oscillation and put them in a list
    max_list = argrelmax(solution[:,0]) 
    min_list = argrelmin(solution[:,0])
    
    #Gets the last min and max x coordinate to find the amplitude
    x_max_point = max_list[0][-1]
    x_min_point = min_list[0][-1]
    #Brings in the x coordinate before and grabs the corresponding y value
    y_max_point = solution[x_max_point,0]
    y_min_point = solution[x_min_point,0]
    #calculates the amplitude by finding the average of min and max
    amplitude = (1.0/2.0)*(y_max_point - y_min_point)

    #Appending the amplitude and theta_res into their lists
    amplitude_list_1.append(amplitude)
    theta_res_list_1.append(theoretical_theta)
    
    
    ###For q = 0.4
    
    #Creates time list depending on what the value of omega_d.
    DDP_Pen = Pendulum(1.0,0.4,0.1,i)
    #Grabs the theoretical theta_res value within the Pendulum class
    theoretical_theta = DDP_Pen.resonance_amp
    solution = DDP_Pen.ode_integrator(p,time)
    
    #Get the min and max of the oscillation and put them in a list
    max_list = argrelmax(solution[:,0]) 
    min_list = argrelmin(solution[:,0])
    
    #Gets the last min and max x coordinate to find the amplitude
    x_max_point = max_list[0][-1]
    x_min_point = min_list[0][-1]
    #Brings in the x coordinate before and grabs the corresponding y value
    y_max_point = solution[x_max_point,0]
    y_min_point = solution[x_min_point,0]
    #calculates the amplitude by finding the average of min and max
    amplitude = (1.0/2.0)*(y_max_point - y_min_point)

    #Appending the amplitude and theta_res into their lists
    amplitude_list_2.append(amplitude)
    theta_res_list_2.append(theoretical_theta)
    
    ###For q = 0.5
    
    #Creates time list depending on what the value of omega_d.
    DDP_Pen = Pendulum(1.0,0.5,0.1,i)
    #Grabs the theoretical theta_res value within the Pendulum class
    theoretical_theta = DDP_Pen.resonance_amp
    solution = DDP_Pen.ode_integrator(p,time)
    
    #Get the min and max of the oscillation and put them in a list
    max_list = argrelmax(solution[:,0]) 
    min_list = argrelmin(solution[:,0])
    
    #Gets the last min and max x coordinate to find the amplitude
    x_max_point = max_list[0][-1]
    x_min_point = min_list[0][-1]
    #Brings in the x coordinate before and grabs the corresponding y value
    y_max_point = solution[x_max_point,0]
    y_min_point = solution[x_min_point,0]
    #calculates the amplitude by finding the average of min and max
    amplitude = (1.0/2.0)*(y_max_point - y_min_point)

    #Appending the amplitude and theta_res into their lists
    amplitude_list_3.append(amplitude)
    theta_res_list_3.append(theoretical_theta)


plt.figure()
plt.plot(omega_driving,amplitude_list_1 , color = "orangered", Label = "$Amplitude$")
plt.plot(omega_driving,theta_res_list_1, color = "darkred", Label = " $ \omega _{res} $",marker = "x", linestyle = "--")
plt.legend(loc= "upper left")
plt.title("When q = 0.2: Amplitude vs $ \omega _{d} $" )
plt.xlabel(" $ \omega _{d} $")
plt.ylabel("Amplitude")
plt.grid() 
plt.show()

plt.figure()
plt.plot(omega_driving,amplitude_list_2 , color = "deepskyblue", Label = "$Amplitude$")
plt.plot(omega_driving,theta_res_list_2, color = "mediumblue", Label = " $ \omega _{res} $",marker = "x", linestyle = "--")
plt.legend(loc= "upper left")
plt.title("When q = 0.4: Amplitude vs $ \omega _{d} $" )
plt.xlabel(" $ \omega _{d} $")
plt.ylabel("Amplitude")
plt.grid() 
plt.show()

plt.figure()
plt.plot(omega_driving,amplitude_list_3 , color = "springgreen", Label = "$Amplitude$")
plt.plot(omega_driving,theta_res_list_3, color = "forestgreen", Label = " $ \omega _{res} $",marker = "x", linestyle = "--")
plt.legend(loc= "upper left")
plt.title("When q = 0.5: Amplitude vs $ \omega _{d} $" )
plt.xlabel(" $ \omega _{d} $")
plt.ylabel("Amplitude")
plt.grid() 
plt.show()


plt.figure()
plt.plot(omega_driving,amplitude_list_1 , color = "orangered", Label = "$Amplitude$ q=0.2")
plt.plot(omega_driving,theta_res_list_1, color = "darkred", Label = " $ \omega _{res} q=0.2$",marker = "x", linestyle = "--")
plt.plot(omega_driving,amplitude_list_2 , color = "deepskyblue", Label = "$Amplitude q=0.4$")
plt.plot(omega_driving,theta_res_list_2, color = "mediumblue", Label = " $ \omega _{res} q=0.4$",marker = "x", linestyle = "--")
plt.plot(omega_driving,amplitude_list_3 , color = "springgreen", Label = "$Amplitude$ q=0.5")
plt.plot(omega_driving,theta_res_list_3, color = "forestgreen", Label = " $ \omega _{res} q=0.5$",marker = "x", linestyle = "--")
plt.legend(loc= "upper left")
plt.title("All 3 q's: Amplitude vs $ \omega _{d} $" )
plt.xlabel(" $ \omega _{d} $")
plt.ylabel("Amplitude")
plt.grid() 
plt.show()



'''












    
    
