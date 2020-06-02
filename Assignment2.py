# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:34:32 2020

@author: joshu
"""

class Date(object):
    """
    This is a class for mathematical operations on calendar dates 
    (from the Gregorian calendar).
    """

    # These methods go back and forth between day/month/year and Julian Day Number
    # note the use of integer division //
    @staticmethod
    def JulianDate(day, month, year):
        """
        Converts a date expressed as day, month, year 
        into a Julian day number
        
        Args:
            day (int): day numbered 1-31
            month (int): month 1-12
            year (int): year
        
        Returns:
            int: Julian Day Number
        """

        a = (14 - month) // 12
        y = year + 4800 - a
        m = month + 12 * a - 3
        return day + (153 * m + 2) // 5 + 365 * y + (y // 4) - (y // 100) + (
            y // 400) - 32045
      
    
    @staticmethod
    def CalendarDate(JDN):

        a = JDN + 32044
        b = (4 * a + 3) // 146097
        c = a - (146097 * b) // 4
        d = (4 * c + 3) // 1461
        e = c - (1461 * d) // 4
        m = (5 * e + 2) // 153
        day = e - ((153 * m + 2) // 5) + 1
        month = m + 3 - 12 * (m // 10)
        year = 100 * b + d - 4800 + (m // 10)
        
        return day, month, year
    
    # class variables that you will need below
    month_names = [
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December"
        ]
    
    day_names = [
            "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday"
        ]
        
    def __init__(self, day, month, year):
        """
        Takes in dates as day,month,year and also the Julian Date Number

        day
        month
        year
        jdn

        """
        #day,month,year are data attributes
        self.day = day
        self.month = month                
        self.year = year
        self.jdn = Date.JulianDate(day, month, year)
    

    def __str__(self):
        """
        Returns a string containing the days of the week, day, month and year
        
        """
        
        
        #This formula finds the days of the week given the Julian Day Number
        dow = (Date.JulianDate(self.day, self.month, self.year) + 1)  % 7
        
        
        outstring = str(self.day_names[dow])
        outstring += " " + str(self.day)
        outstring += " " + str(self.month_names[self.month - 1])
        outstring += " " + str(self.year)
        return outstring
    
    def __add__(self,other):
        if isinstance(other, (float,int)):
            new_date = self.jdn + other
            new_calendar = Date.CalendarDate(new_date)
            return Date(new_calendar[0],new_calendar[1],new_calendar[2])
        else:
            return Date.JulianDate(self.day, self.month, self.year) + Date.JulianDate(other.day, other.month, other.year)

    
    def __sub__(self,other):
        if isinstance(other, (float,int)):
            new_date_sub = self.jdn - other
            new_calendar_sub = Date.CalendarDate(new_date_sub)
            return Date(new_calendar_sub[0],new_calendar_sub[1],new_calendar_sub[2])
        else:
            return Date.JulianDate(self.day, self.month, self.year) - Date.JulianDate(other.day, other.month, other.year)



'''
today = Date(22,1,2020)
print("Today,",today,"has Julian Day number", today.jdn)

tomorrow = today + 1
print("Tomorrow will be",str(tomorrow)+".")
print("There is",tomorrow-today,"day between tomorrow and today.")


leafs_cup=Date(2,5,1967)
print("The Toronto Maple Leafs last won the Stanley Cup on", leafs_cup,".")
print("It has been", today-leafs_cup,"days since then ....")

        
#####



Birthday = Date(29,6,1997) 
today =  Date(22,1,2020) 
print("You are ", today.jdn - Birthday.jdn , " days old")    

 

#####


#Array of 1000 year gaps from 1000 to 30000 inclusive.
year_gaps = range(1000,31000,1000)
birthdate_gaps = []


for i in range(0,len(year_gaps)):
    a = Birthday + year_gaps[i]
    birthdate_gaps.append(str(a))


print (birthdate_gaps)

'''


#####


class AmericanDate(Date):
    """ AmericanDate is a subclass of Date. 

    data attributes:

    day
    month
    year
    jdn
    
    """
    
    def __init__(self, day,month,year):
        # start AmericanDate __init__ by calling the Date __init__ function
        Date.__init__(self,day,month,year)
        

    def __str__(self):
        """
        Returns a string containing the days of the week, day, month and year
        
        """
        
        
        #This formula finds the days of the week given the Julian Day Number
        dow = (Date.JulianDate(self.day, self.month, self.year) + 1)  % 7
        
        #Exact copy of Date's __str__ but the order of Month and Day is switched
        outstring = str(self.day_names[dow])
        outstring += " " + str(self.month_names[self.month - 1])
        outstring += " " + str(self.day)
        outstring += " " + str(self.year)
        return outstring

independence_day = AmericanDate(4,7,1776)
print(independence_day)







