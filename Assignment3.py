# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:05:01 2020

@author: joshu
"""


import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from math import sqrt




class point:
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

class body:
    def __init__(self, location, mass, velocity, name = ""):
        self.location = location
        self.mass = mass
        self.velocity = velocity
        self.name = name
        
star1 = {"location":point(-7.48e10,-4.31844e10,0), "mass":1.989e30, "velocity":point(14894.68,-25798.34,0)}
star2 = {"location":point(0,8.637033e10,0), "mass":1.989e30, "velocity":point(-29789.36,0,0)}
star3 = {"location":point(7.48e10,4.31844e10,0), "mass":1.989e30, "velocity":point(14894.68,25798.34,0)}



bodies = [
    body( location = star1["location"], mass = star1["mass"], velocity = star1["velocity"], name = "Star 1"),
    body( location = star2["location"], mass = star2["mass"], velocity = star2["velocity"], name = "Star 2"),
    body( location = star3["location"], mass = star3["mass"], velocity = star3["velocity"], name = "Star 3"),
    ]


body_locations_hist = []
body_velocity_hist = []
other_bodies_index = range(0,len(bodies))

t0 = 0
t1 = 2e7
N = 33000
h = (t1-t0)/N
time = np.arange(t0,t1+h,h)

#Creating list for position and velocity for each body
for current_body in bodies:
    body_locations_hist.append({"x":[], "y":[], "z":[], "name":current_body.name})
    body_velocity_hist.append({"x":[], "y":[], "z":[], "name":current_body.name})
 

#Adding in the intial conditions of position and velocity as the 0th index of the list.
for i in range(0,len(bodies)):
    body_locations_hist[i]['x'].append(bodies[i].location.x)
    body_locations_hist[i]['y'].append(bodies[i].location.y)
    body_locations_hist[i]['z'].append(bodies[i].location.z)
    body_velocity_hist[i]['x'].append(bodies[i].velocity.x)
    body_velocity_hist[i]['y'].append(bodies[i].velocity.y)
    body_velocity_hist[i]['z'].append(bodies[i].velocity.z)

#Eulers method for calculating acceleration
def acceleration_euler(tar_bodies_x, other_bodies_x,tar_bodies_y, other_bodies_y,tar_bodies_z, other_bodies_z,other_mass):
            G_const = 6.67408e-11
            d = (other_bodies_x - tar_bodies_x)**2 + (other_bodies_y - tar_bodies_y)**2 + (other_bodies_z - tar_bodies_z)**2
            d = sqrt(d) 
            force = G_const * other_mass / d**3
            acceleration.x = acceleration.x + force*(other_bodies_x - tar_bodies_x)
            acceleration.y = acceleration.y + force*(other_bodies_y - tar_bodies_y)
            acceleration.z = acceleration.z + force*(other_bodies_z - tar_bodies_z)
            return acceleration

#Used for calculating the temperory step for velocity and position for RK
def temp_step(point1, point2, h):
    ret = point1 + point2 * h
    return ret


        
def acceleration_RK4(tar_bodies_x, other_bodies_x,tar_bodies_y, other_bodies_y,tar_bodies_z, other_bodies_z,other_mass, tar_bodies_vel_x,tar_bodies_vel_y,tar_bodies_vel_z):
            G_const = 6.67408e-11
            #Equation for Gravity         
            d = (other_bodies_x - tar_bodies_x)**2 + (other_bodies_y - tar_bodies_y)**2 + (other_bodies_z - tar_bodies_z)**2
            d = sqrt(d) 
            force = G_const * other_mass / d**3
            
            #1st Order Runge-Kutta. Same as Euler.
            k1.x = force * (other_bodies_x - tar_bodies_x )
            k1.y = force * (other_bodies_y - tar_bodies_y)
            k1.z = force * (other_bodies_z - tar_bodies_z )
            
            #Grabbing the temperory velocity of the particle when we shift it up on the 2nd order RK
            tmp_vel_x = temp_step(tar_bodies_vel_x, k1.x, 0.5)
            tmp_vel_y = temp_step(tar_bodies_vel_y, k1.y, 0.5)
            tmp_vel_z = temp_step(tar_bodies_vel_z, k1.z, 0.5)
            
            #Grabbing the temperory position of the particle when we shift it up on the 2nd order RK
            tmp_loc_x = temp_step(tar_bodies_x, tmp_vel_x, 0.5)
            tmp_loc_y = temp_step(tar_bodies_y, tmp_vel_y, 0.5)
            tmp_loc_z = temp_step(tar_bodies_z, tmp_vel_z, 0.5)

	    	#2nd Order Runge-Kutta
            k2.x = (other_bodies_x - (tmp_loc_x + tmp_vel_x * 0.5 * h)) * force
            k2.y = (other_bodies_y - (tmp_loc_y + tmp_vel_y * 0.5 * h)) * force
            k2.z = (other_bodies_z - (tmp_loc_z + tmp_vel_z * 0.5 * h)) * force
            
            #Grabbing the temperory velocity of the particle when we shift it up on the 3nd order RK
            tmp_vel_x = temp_step(tar_bodies_vel_x, k2.x, 0.5)
            tmp_vel_y = temp_step(tar_bodies_vel_y, k2.y, 0.5)
            tmp_vel_z = temp_step(tar_bodies_vel_z, k2.z, 0.5)

	    	#3nd Order Runge-Kutta
            k3.x = (other_bodies_x - (tmp_loc_x + tmp_vel_x * 0.5 * h)) * force
            k3.y = (other_bodies_y - (tmp_loc_y + tmp_vel_y * 0.5 * h)) * force
            k3.z = (other_bodies_z - (tmp_loc_z + tmp_vel_z * 0.5 * h)) * force
            
            #Grabbing the temperory velocity of the particle when we shift it up on the 4nd order RK
            tmp_vel_x = temp_step(tar_bodies_vel_x, k3.x, 1)
            tmp_vel_y = temp_step(tar_bodies_vel_y, k3.y, 1)
            tmp_vel_z = temp_step(tar_bodies_vel_z, k3.z, 1)
            
            #Grabbing the temperory position of the particle when we shift it up on the 4nd order RK
            tmp_loc_x = temp_step(tar_bodies_x, tmp_vel_x, 0.5)
            tmp_loc_y = temp_step(tar_bodies_y, tmp_vel_y, 0.5)
            tmp_loc_z = temp_step(tar_bodies_z, tmp_vel_z, 0.5)

	    	#4nd Order Runge-Kutta
            k4.x = (other_bodies_x - (tmp_loc_x + tmp_vel_x * h)) * force
            k4.y = (other_bodies_y - (tmp_loc_y + tmp_vel_y * h)) * force
            k4.z = (other_bodies_z - (tmp_loc_z + tmp_vel_z * h)) * force
            

            #Calculating the accelerations
            acceleration.x = acceleration.x + (k1.x + k2.x * 2 + k3.x * 2 + k4.x) / 6
            acceleration.y = acceleration.y + (k1.y + k2.y * 2 + k3.y * 2 + k4.y) / 6
            acceleration.z = acceleration.z + (k1.z + k2.z * 2 + k3.z * 2 + k4.z) / 6
            
            '''
            print (k1.y)
            print(other_bodies_y)
            print(tar_bodies_y)
            print(force)
            print (k2.y)
            print (k3.y)
            print (k4.y)
            print("next")
            '''
            
            return acceleration
        
def acceleration_RK2(tar_bodies_x, other_bodies_x,tar_bodies_y, other_bodies_y,tar_bodies_z, other_bodies_z,other_mass, tar_bodies_vel_x,tar_bodies_vel_y,tar_bodies_vel_z):
            G_const = 6.67408e-11
            #Equation for Gravity
            d = (other_bodies_x - tar_bodies_x)**2 + (other_bodies_y - tar_bodies_y)**2 + (other_bodies_z - tar_bodies_z)**2
            d = sqrt(d) 
            force = G_const * other_mass / d**3
            
            #1st Order Runge-Kutta. Same as Euler.
            k1.x = force * (other_bodies_x - tar_bodies_x )
            k1.y = force * (other_bodies_y - tar_bodies_y )
            k1.z = force * (other_bodies_z - tar_bodies_z )
            
            #Grabbing the temperory velocity of the particle when we shift it up on the 2nd order RK
            tmp_vel_x = temp_step(tar_bodies_vel_x, k1.x, 0.5)
            tmp_vel_y = temp_step(tar_bodies_vel_y, k1.y, 0.5)
            tmp_vel_z = temp_step(tar_bodies_vel_z, k1.z, 0.5)
            
            #Grabbing the temperory position of the particle when we shift it up on the 2nd order RK
            tmp_loc_x = temp_step(tar_bodies_x, tmp_vel_x, 0.5)
            tmp_loc_y = temp_step(tar_bodies_y, tmp_vel_y, 0.5)
            tmp_loc_z = temp_step(tar_bodies_z, tmp_vel_z, 0.5)
            

	    	#2nd Order Runge-Kutta
            k2.x = (other_bodies_x - (tmp_loc_x + tmp_vel_x * 0.5 * h)) * force
            k2.y = (other_bodies_y - (tmp_loc_y + tmp_vel_y * 0.5 * h)) * force
            k2.z = (other_bodies_z - (tmp_loc_z + tmp_vel_z * 0.5 * h)) * force
            
            #Calculating the accelerations
            acceleration.x =  acceleration.x + (k1.x + k2.x) / 2.0
            acceleration.y = acceleration.y + (k1.y + k2.y)  / 2.0
            acceleration.z = acceleration.z + (k1.z + k2.z)  /2.0
            
            
            return acceleration
            
#Updates the velocity of the particle
def update_velocity(acc_x,acc_y,acc_z):
        velocity = point(0,0,0)
        velocity.x = acc_x * h
        velocity.y = acc_y * h
        velocity.z = acc_z * h
        return velocity

#Updates the position of the particle
def update_location(vel_x,vel_y,vel_z):
    position = point(0,0,0)
    position.x = vel_x * h
    position.y = vel_y * h
    position.z = vel_z * h
    return position



#This loops runs for whoever many timesteps the user wants. 
for i in range(0,N):  
    #Resets the acceleration and R-K orders for each particle
    acceleration = point(0,0,0) 
    k1 = point (0,0,0)
    k2 = point (0,0,0)
    k3 = point (0,0,0)
    k4 = point (0,0,0)
    tmp_loc = point (0,0,0)
    tmp_vel = point (0,0,0)
    
    #This loops over the target particle that feels gravity from all other particles. Has index labelled tar_index.
    for tar_index,bodies_info in enumerate(bodies):
        tar_bodies = bodies[tar_index]
        acceleration = point(0,0,0)
        #This loops over all the other particles so we can calculate the force they have on our target particle
        for other_index in other_bodies_index:
            other_bodies = bodies[other_index]
            #Makes sure that our target particle and the other particles are different. So we wont find the force of the particle on itself.
            if tar_index != other_index:
                
                #Different Methods of calculating. Un-comment to use whichever method
                #acc = (acceleration_euler(body_locations_hist[tar_index]['x'][i],body_locations_hist[other_index]['x'][i],body_locations_hist[tar_index]['y'][i],body_locations_hist[other_index]['y'][i],body_locations_hist[tar_index]['z'][i],body_locations_hist[other_index]['z'][i], other_bodies.mass))
                #acc = (acceleration_RK2(body_locations_hist[tar_index]['x'][i],body_locations_hist[other_index]['x'][i],body_locations_hist[tar_index]['y'][i],body_locations_hist[other_index]['y'][i],body_locations_hist[tar_index]['z'][i],body_locations_hist[other_index]['z'][i], other_bodies.mass, body_velocity_hist[tar_index]['x'][i], body_velocity_hist[tar_index]['y'][i], body_velocity_hist[tar_index]['z'][i]))
                acc = (acceleration_RK4(body_locations_hist[tar_index]['x'][i],body_locations_hist[other_index]['x'][i],body_locations_hist[tar_index]['y'][i],body_locations_hist[other_index]['y'][i],body_locations_hist[tar_index]['z'][i],body_locations_hist[other_index]['z'][i], other_bodies.mass, body_velocity_hist[tar_index]['x'][i], body_velocity_hist[tar_index]['y'][i], body_velocity_hist[tar_index]['z'][i]))

                #Call the update_velocity function and also append all the velocity to list
        vel = update_velocity(acc.x,acc.y,acc.z)
        body_velocity_hist[tar_index]['x'].append(vel.x + body_velocity_hist[tar_index]['x'][i])
        body_velocity_hist[tar_index]['y'].append(vel.y + body_velocity_hist[tar_index]['y'][i])
        body_velocity_hist[tar_index]['z'].append(vel.z + body_velocity_hist[tar_index]['z'][i])
            
            #Call the update_location function and also append all the positions to list
        updated_loc = update_location(vel.x + body_velocity_hist[tar_index]['x'][i],vel.y+ body_velocity_hist[tar_index]['y'][i],vel.z+ body_velocity_hist[tar_index]['z'][i]) 
        body_locations_hist[tar_index]['x'].append(updated_loc.x + body_locations_hist[tar_index]['x'][i])
        body_locations_hist[tar_index]['y'].append(updated_loc.y + body_locations_hist[tar_index]['y'][i])
        body_locations_hist[tar_index]['z'].append(updated_loc.z + body_locations_hist[tar_index]['z'][i])







plt.figure()
plt.plot(body_locations_hist[0]['x'], body_locations_hist[0]['y'], label="Star 1",color = "gold",linestyle = "dashed")
plt.plot(body_locations_hist[1]['x'], body_locations_hist[1]['y'], label="Star 2",color = "skyblue",marker='+')
plt.plot(body_locations_hist[2]['x'], body_locations_hist[2]['y'], label="Star 3",color = "lawngreen",marker='x')
plt.xlabel("X-Axis (m)")
plt.ylabel('Y-Axis(m)')
plt.title("Motion of 3-Body System" )
plt.legend(loc="best")
plt.grid()


'''
print (body_locations_hist[0]['x'][-1])
print (body_locations_hist[0]['y'][-1])
print (body_velocity_hist[0]['x'][-1])
print (body_velocity_hist[0]['y'][-1])

print (body_locations_hist[1]['x'][-1])
print (body_locations_hist[1]['y'][-1])
print (body_velocity_hist[1]['x'][-1])
print (body_velocity_hist[1]['y'][-1])

print (body_locations_hist[2]['x'][-1])
print (body_locations_hist[2]['y'][-1])
print (body_velocity_hist[2]['x'][-1])
print (body_velocity_hist[2]['y'][-1])
'''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
