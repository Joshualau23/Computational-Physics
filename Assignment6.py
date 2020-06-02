# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:05:21 2020

@author: joshu
"""

from math import e
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt



#Defining all the constants
N_0 = 1000
k = 1.0
#delta_t = 0.00001
delta_t = 0.001
stop_time = 2.0
time = np.arange(0,stop_time, delta_t)

def decay_solution(t):
    return N_0*e**(-k*t)

# Define mu
mean_decays = k*decay_solution(time)*delta_t

# Create a list to store all the number of decays from poission distribution
# For the most part the values of this list is 0 or 1 but you might get 2 sometimes
decays_list = np.zeros_like(time)

for i in range(0,len(time)):
    #Generates a random number from the poisson distribution
    r = poisson.rvs(mean_decays[i], size=1)
    decays_list[i] = r[0]

# I used the poisson distribution package from scipy as a result therefore the number of decays
# won't be strictly 0 or 1, there is a small chnace you would see 2 decays at the beginning.
plt.plot(time,decays_list, color = "gold")
plt.title("Number of Decays vs Time" )
plt.xlabel("Time")
plt.ylabel("Number of Decays")
plt.grid()   
plt.show()


###b)

from decimal import Decimal
from math import sqrt

#Defining new rebining delta t
new_delta_t  = 0.1
slices_list = []

#Finds the modulus of the new delta t into the original end time
#This is for when the end time can't be evenly divided by the new delta t
float_remainder = float(round(Decimal(stop_time) % Decimal(new_delta_t),5))

#if statement for whether the float remainder is a 0 or something else

# if it can be divived into equal parts then the number of bins is simply
# just the end time divided by the new delta t
# slices_list gives the indices for the number of decay list thats to be sliced
if float_remainder == new_delta_t:
    number_of_bins = (stop_time) / new_delta_t
    number_of_bins = int(number_of_bins)
    slices_list = np.linspace(0,len(time), number_of_bins + 1,endpoint = True)
    
# if it can't be divived into equal parts then the number of bins is simply
# just the end time divided by the new delta t + 1 where the 1 represents all
# the indices of the number of decay list that are left over
else:
    number_of_bins = (stop_time - float_remainder) / new_delta_t + 1.0
    number_of_bins = int(number_of_bins)
    slices_list = np.linspace(0,len(time), number_of_bins + 1,endpoint = True)

#rounding down the numbers since indices must be integers 
slices_list = np.around(slices_list,0)
#Creating the new rebinned time list
new_time = np.arange(0,stop_time, new_delta_t)
new_decays_list = []
error_list= []

#Loop for putting the correct indices into the number of decay list so they can
# be rebinned to a new smaller list.
for i in range(0,len(slices_list)):
    if i == 0:
        continue
    elif i == len(slices_list):
        rebinning_list = decays_list[int(slices_list[i]):-1]
    else:
        rebinning_list = decays_list[int(slices_list[i-1]):int(slices_list[i])]
    #Summing up all the values within each bin
    rebinning = np.sum(rebinning_list)
    #Getting the error of each bin
    error = sqrt(rebinning)
    #Appending the values to their respective list
    new_decays_list.append(rebinning)
    error_list.append(error)


#print (new_decays_list)
#print (error_list)

plt.errorbar(new_time,new_decays_list,yerr = error_list, color = "deepskyblue",ecolor="red")
plt.title("Number of Decays vs Time (Rebinned)" )
plt.xlabel("Time")
plt.ylabel("Number of Decays")
plt.grid()   
plt.show()


###c)


#Takes the natural log of the new decay list
log_number_of_decays = np.log(new_decays_list)

#The uncertainty when taking the natural log of a a value is simply delta_x / x
log_error_list = np.array(error_list) / np.array(new_decays_list)
#print (log_error_list)


line_bestfit,error_matrix = np.polyfit(new_time,log_number_of_decays,1,cov=True)
error = np.sqrt(np.diag(error_matrix))

slope_bestfit = line_bestfit[0]
slope_error = error[0]
b_bestfit = line_bestfit[1]
b_error = error[1]

bestfit_list = slope_bestfit*new_time + b_bestfit


plt.errorbar(new_time,log_number_of_decays,yerr = log_error_list, color = "deepskyblue",ecolor="red",label = "Data")
plt.plot(new_time,bestfit_list,color = "gold", label = "Best Fit")
plt.title("Number of Decays vs Time (Rebinned)" )
plt.xlabel("Time")
plt.ylabel("Number of Decays")
plt.legend(loc = "best")
plt.grid()   
plt.show()



print ("The best fit constant are: " + "For slope " + str(slope_bestfit) + " +/- " + str(slope_error) + " For intercept " + str(b_bestfit) + " +/- " + str(b_error))













