# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:12:58 2020

@author: joshu
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.signal import argrelmin, argrelmax

plt.rcParams['figure.figsize'] = [12, 6] # makes the default plot size larger



class Periodic(object):
    """
    This class that takes in periodic data and outputs Fourier Transform of said data.
    
    Data Attributes:
        data
        h
        
    
    Methods:
        dft - Calculates the Discrete Fourier Transform of data
        fft - Uses the fft package from scipy.fft
        fftfreq - Uses fftfreq package from scipy.fft
        fftangfreq - Calculates the angular frequency of each Fourier Transformed data element
    """
    def __init__(self,data,h):
        """
        DDP Pendulum
        
        Parameters:
        -----------
        data
        h 
        """

        self.data = data         
        self.h = h 
        
        
    def dft(self):
        N = len(self.data) 
        n = np.arange(0,N)
        k = n.reshape((N, 1))
        Transform = np.exp(-1j*(2 * np.pi * k * n) / N)
        return np.dot(self.data,Transform)

    def fft(self):
        return np.fft.fft(self.data)
    
    def fftfreq(self):
        self.data = np.asarray(self.data, dtype=float)
        window_length = self.data.size
        return np.fft.fftfreq(window_length,self.h)
    
    def fftangfreq(self):
        angfreq = 2 * np.pi * Periodic.fftfreq(self)
        return angfreq

'''
test_list = [1,2,4,5,6,7,8]
test = Periodic(test_list, 0.1)
print (test.fft())
print (test.fftfreq())
print (test.fftangfreq())
'''

class Pendulum(object):
    """
    Simulate a damped driven pendulum in phase space
    """

    def __init__(self, omega_0=1., q=0., A=0., omega_d=0.):
        """
        Args:
            omega_0 : Natural angular frequency
            q       : damping factor q
            A       : Amplitude of driving force
            omega_d : Driving angular frequency
            
        """
        self.omega_0 = omega_0 # Natural angular frequency
        self.q = q             # damping factor q
        self.A = A             # Amplitude of driving force
        self.omega_d = omega_d # Driving angular frequency
        
        self.period_0 = 2*np.pi/self.omega_0
        if self.omega_d > 0:
            self.period_d = 2.*np.pi/self.omega_d
        else:
            self.period_d = np.inf
        self.resonance_amp = A/np.sqrt((omega_0**2-omega_d**2)**2+q**2*omega_d**2)
        
    def acc(self, p, t):
        """
        Calculate angular acceleration
        
        Args:
            p: phase space coordinates
            t: time
            
        Returns:
            angular acceleration        
        """
        
        a = -(self.omega_0**2)*np.sin(p[0])  # basic (non-linear) pendulum
        a += -self.q*p[1]                    # add damping
        a += self.A*np.cos(self.omega_d*t)   # add driving force
        return a
    
    def dpdt(self,p,t):
        """
        Return phase space derivatives to be used by odeint
        """
        return np.array([p[1], self.acc(p, t)])


#----------------------------------------------------------------------

'''
#Set up a simple pendulum with default parameters and run it for 20 periods

pend = Pendulum()  # construct a pendulum with default parameters

nperiod = 8      # number of periods
nper_period = 1000  # timesteps per period
times = np.arange(0., pend.period_0 * nperiod, pend.period_0 / nper_period)

# initial condition: small initial angle, zero initial angular velocity
theta_0 = 1. * np.pi / 180.
thetadot_0 = 0.

# Run the pendulum with odeint and get the output
output = odeint(pend.dpdt, [theta_0, thetadot_0], times)

'''

'''
# theta coordinate lives in column 0 of the output
plt.plot(times, output[:, 0])

# draw vertical lines
for i in np.arange(0, nperiod + 1):
    plt.axvline(pend.period_0 * i, ls=":")

'''


'''

#### b)



time_step = pend.period_0 / nper_period
test_case = Periodic(output[:, 0],time_step)
discrete_ft = test_case.dft()
angular_freq = test_case.fftangfreq()

#print (len(discrete_ft))
#print (len(angular_freq))


#for i in discrete_ft:
#    if i.real > 1e-1:
#        print (i)
#    else:
#        continue


plt.plot(angular_freq,discrete_ft, "o")
plt.title("Simple Harmonic Pendulum" )
plt.xlabel("Angular Frequency")
plt.ylabel("Fourier Transform")
plt.grid()   
plt.show()



### c)


fast_ft = test_case.fft()


a = 0
for i in fast_ft:
    a += 1
    if i.real > 1e-1:
        print (i)
        print (angular_freq[a])
    else:
        continue


plt.plot(angular_freq,fast_ft, "o")
plt.title("Simple Harmonic Pendulum" )
plt.xlabel("Angular Frequency")
plt.ylabel("Fourier Transform")
plt.grid()   
plt.show()

'''





### d)

'''
#Creating all 3 q's

Q1 = 1.34
q1 = 1/ Q1

Q2 = 1.36
q2 = 1 / Q2

Q3 = 1.3745 
q3 = 1 / Q3


   

#Creating 3 pendulum with the parameters given in the question.
pend1 = Pendulum(1., q1,1.5, 2.0/3.0)  
pend2 = Pendulum(1., q2,1.5, 2.0/3.0)  
pend3 = Pendulum(1., q3,1.5, 2.0/3.0)  

drivingperiod = (2 * np.pi) / (2. / 3.)
nperiod = 50*drivingperiod      # number of periods
nper_period = 500  # timesteps per period
times = np.arange(0., pend1.period_0 * nperiod, pend1.period_0 / nper_period) # WLOG I've used pend1's period since only q is different

# initial condition: small initial angle, zero initial angular velocity
theta_0 = 0.
thetadot_0 = 0.

# Run the 3 pendulums with odeint and get the output
output1 = odeint(pend1.dpdt, [theta_0, thetadot_0], times)
output2 = odeint(pend2.dpdt, [theta_0, thetadot_0], times)
output3 = odeint(pend3.dpdt, [theta_0, thetadot_0], times)


patern_repeat = 10000
#plt.plot(times[:patern_repeat],output1[:patern_repeat, 0], color = "Gold", label = "Q = 1.34")
#plt.plot(times[:patern_repeat],output2[:patern_repeat, 0], color = "mediumblue", label = "Q = 1.36")
plt.plot(times[:patern_repeat],output3[:patern_repeat, 0], color = "limegreen", label = "Q = 1.3745")
plt.title("Driven Damped Pendulum" )
plt.xlabel("Time")
plt.ylabel("Angle")
plt.legend(loc="best")
plt.grid()   
plt.show()



#Starting point where the last 5 period begins
starting_point = 212085

#Discarding all the data up to the 45 period.Saving the last 5 period.
output1 = output1[212085:, 0]
output2 = output2[212085:, 0]
output3 = output3[212085:, 0]

#Finding the min and max of 3 graphs, since Q = 1.3745 is the highest of the 3 cases.
ylo = min(output3)
yhi = max(output3)

plt.plot(times[212085:],output1, color = "Gold", label = "Q = 1.34")
plt.plot(times[212085:],output2, color = "mediumblue", label = "Q = 1.36")
plt.plot(times[212085:],output3, color = "limegreen", label = "Q = 1.3745")
plt.title("Driven Damped Pendulum" )
plt.xlabel("Time")
plt.ylabel("Angle")
plt.legend(loc="best")
plt.ylim(ylo, yhi)
plt.grid()   
plt.show()




#Again since the period should be the same i've used pend1 for the timestep
time_step = pend1.period_0 / nper_period

#For Q = 1.34
case1 = Periodic(output1,time_step) #Plugs output1 data into Periodic class
fast_ft1 = case1.fft()              #Grabs the fft method from Periodic class and perfroms it
angular_freq1 = case1.fftangfreq()  #Uses fftangfreq method
index1 = np.where(np.around(angular_freq1,2) == 2.00) #Finds the index where the angular frequency array is equal 2
#print (angular_freq1[index1[0][0]])


#For Q = 1.36
case2 = Periodic(output2,time_step) #Plugs output1 data into Periodic class
fast_ft2 = case2.fft()              #Grabs the fft method from Periodic class and perfroms it
angular_freq2 = case2.fftangfreq()  #Uses fftangfreq method
index2 = np.where(np.around(angular_freq2,2) == 2.00) #Finds the index where the angular frequency array is equal 2
#print (angular_freq2[index2[0][0]])

#For Q = 1.3745
case3 = Periodic(output3,time_step) #Plugs output1 data into Periodic class
fast_ft3 = case3.fft()              #Grabs the fft method from Periodic class and perfroms it
angular_freq3 = case3.fftangfreq()  #Uses fftangfreq method
index3 = np.where(np.around(angular_freq3,2) == 2.00) #Finds the index where the angular frequency array is equal 2
#print (angular_freq3[index1[0][0]])

plt.plot(angular_freq1[1:index1[0][0]],fast_ft1[1:index1[0][0]], "o" ,color = "Gold", label = "Q = 1.34")
plt.plot(angular_freq2[1:index2[0][0]],fast_ft2[1:index2[0][0]], "o" ,color = "mediumblue", label = "Q = 1.36")
plt.plot(angular_freq3[1:index3[0][0]],fast_ft3[1:index3[0][0]], "o" ,color = "limegreen", label = "Q = 1.3745")
plt.title("Discrete Fourier Transform vs. Angular Frequency" )
plt.xlabel("Angular Frequency")
plt.ylabel("DFT")
plt.legend(loc="best")
plt.grid()   
plt.show()

#Finds what the frequency is at the large peaks
maxfreq = max(fast_ft1[1:index1[0][0]])
max_index = np.where( fast_ft1[1:index1[0][0]] == maxfreq) 
print (angular_freq3[max_index[0][0]])

'''






###e)


'''

Q1 = 1.3745
q1 = 1/ Q1

Q2 = 1.3758
q2 = 1 / Q2



#Creating 2 pendulum with the parameters given in the question.
pend1 = Pendulum(1., q1,1.5, 2.0/3.0)  
pend2 = Pendulum(1., q2,1.5, 2.0/3.0)  


drivingperiod = (2 * np.pi) / (2. / 3.)
nperiod = 1024*drivingperiod      # number of periods
nper_period = 1000  # timesteps per period
times = np.arange(0., pend1.period_0 * nperiod, pend1.period_0 / nper_period) # WLOG I've used pend1's period since only q is different



# initial condition: small initial angle, zero initial angular velocity
theta_0 = 0.
thetadot_0 = 0.

# Run the 3 pendulums with odeint and get the output
output1 = odeint(pend1.dpdt, [theta_0, thetadot_0], times)
output2 = odeint(pend2.dpdt, [theta_0, thetadot_0], times)



#Starting point where the last 5 period begins
starting_point = (1024 - 128)*(len (np.arange(0., pend1.period_0 * nperiod / 1024., pend1.period_0 / nper_period)))

print(starting_point)

#Discarding all the data up to the 45 period.Saving the last 5 period.
output1 = output1[starting_point:, 0]
output2 = output2[starting_point:, 0]


#Again since the period should be the same i've used pend1 for the timestep
time_step = pend1.period_0 / nper_period

#For Q = 1.3745
case1 = Periodic(output1,time_step) #Plugs output1 data into Periodic class
fast_ft1 = case1.fft()              #Grabs the fft method from Periodic class and perfroms it
angular_freq1 = case1.fftangfreq()  #Uses fftangfreq method
index1 = np.where(np.around(angular_freq1,3) == 0.10) #Finds the index where the angular frequency array is equal 2
#print (angular_freq1[index1[0][0]])


#For Q = 1.3758
case2 = Periodic(output2,time_step) #Plugs output1 data into Periodic class
fast_ft2 = case2.fft()              #Grabs the fft method from Periodic class and perfroms it
angular_freq2 = case2.fftangfreq()  #Uses fftangfreq method
index2 = np.where(np.around(angular_freq2,3) == 0.040) #Finds the index where the angular frequency array is equal 2
#print (angular_freq2[index2[0][0]])



plt.plot(angular_freq1[1:index1[0][0]],fast_ft1[1:index1[0][0]], "o" ,color = "Gold", label = "Q = 1.34")
#plt.plot(angular_freq2[1:index2[0][0]],fast_ft2[1:index2[0][0]], "o" ,color = "mediumblue", label = "Q = 1.36")
plt.title("Discrete Fourier Transform vs. Angular Frequency" )
plt.xlabel("Angular Frequency")
plt.ylabel("DFT")
plt.legend(loc="best")   
plt.show()


'''





###f)



Q1 = 1.34
q1 = 1/ Q1

#Creating 2 pendulum with the parameters given in the question.
pend1 = Pendulum(1., q1,1.5, 2.0/3.0)  


drivingperiod = (2 * np.pi) / (2. / 3.)
nperiod = 256*drivingperiod      # number of periods
nper_period = 200  # timesteps per period
times = np.arange(0., pend1.period_0 * nperiod, pend1.period_0 / nper_period) # WLOG I've used pend1's period since only q is different



# initial condition: small initial angle, zero initial angular velocity
theta_0 = 0.
thetadot_0 = 0.



# Run the 1 pendulums with odeint and get the output
output1 = odeint(pend1.dpdt, [theta_0, thetadot_0], times)



#Starting point where the last 5 period begins
starting_point = (256 - 128)*(len (np.arange(0., pend1.period_0 * nperiod / 256., pend1.period_0 / nper_period)))



angular_vecloity1 = (output1[starting_point:,1])

#Making Q list where the shape is equal to angular velocity
Q1_list = Q1 * np.ones_like(angular_vecloity1)

print(angular_vecloity1)





plt.plot(Q1_list,angular_vecloity1, "," ,color = "mediumblue", label = "Q = 1.34")
plt.title("Q vs Angular Velocity" )
plt.xlabel("Q")
plt.ylabel("Angular Velocity")
plt.legend(loc="best")   
plt.show()






#Making Q = 1.34 to Q = 1.38 Figure

Q_steps = 0.0005
Q_array = np.arange(1.34,1.38 + Q_steps, Q_steps)

saved_angular_velocity = []
saved_Q = []


for i in Q_array:
    #Setting damping factor
    q = 1.0 / i 
    
    #Creating the pendulum with different values of q
    pend = Pendulum(1., q ,1.5, 2.0/3.0)
    
    #Using odeint to solve pendulum DE
    output = odeint(pend.dpdt, [theta_0, thetadot_0], times)
    
    #Saving the angular velcoity to it's own list
    angular_velocity = (output[starting_point:,1])

    
    #Creating a Q list the same size as angular velcoity
    Q_list = i * (np.ones_like(angular_velocity))
    
    #Adding Each Q and Angular Velocity list to the total saved list
    saved_Q.append(Q_list)
    saved_angular_velocity.append(angular_velocity)
    


plt.figure()
for i in range(len(saved_Q)):
    plt.plot(saved_Q[i],saved_angular_velocity[i], "," ,color = "mediumblue")
    plt.title("Q vs Angular Velocity" )
    plt.xlabel("Q")
    plt.ylabel("Angular Velocity")
plt.show()



