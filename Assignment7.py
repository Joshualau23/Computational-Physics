import numpy as np
import matplotlib.pyplot as plt


class Ising2D(object):
    """    
    Data Attributes:
        size
        temp
        
        
    
    Methods:

    """
    
    def __init__(self,temp=3.,size=10):
        self.temp = temp
        self.size = size     
        self.energy_site = np.zeros((size,size))
        self.spin_site = np.zeros((size,size))
        self.generated_lattice = np.zeros((size,size))
        

    def generate_lattice(self):
        lattice = np.random.randint(2, size=(self.size,self.size))
        lattice = 2*lattice - 1
        return lattice

    def pairs(self,lattice,x,y):
        left   = (x, y - 1)
        right  = (x, (y + 1)% self.size)
        top    = (x - 1, y)
        bot = ((x + 1)% self.size, y)
        spin_array = np.array((lattice[left],lattice[right],lattice[top],lattice[bot]))
        return spin_array
            
    def equilibrium(self, flips_per_site=100):
        #Generates an Ising lattice
        ising_lattice = self.generate_lattice()
        
        #Keeps track of how many flips each component is flipped
        flipped_array = np.zeros_like(ising_lattice)
        """ Allows sytems to come to equilibrium by performing a number of flip attempts per site

        Args:
            flips_per_site : average number of attempts per site        
        """
        

        
        #Keeps track of number of flips
        average_flips = np.sum(flipped_array) / self.size
        while average_flips < flips_per_site:
            #Generates numbers between 0 to size of lattice for x and y
            
            random_x = np.random.randint(0, self.size )
            random_y = np.random.randint(0, self.size )

            #Rembering the old energy value before the flip
            old_energy = self.get_average_energy_per_site(ising_lattice,random_x,random_y)
            
            #Flip Spin
            ising_lattice[random_x,random_y] = -1*ising_lattice[random_x,random_y]
            
            #Calculate new energy
            new_energy = self.get_average_energy_per_site(ising_lattice,random_x,random_y)
            delta_energy = new_energy - old_energy
            if new_energy <= old_energy:
                flipped_array[random_x,random_y] += 1
            else:
                if np.exp((-1.0 * (delta_energy))/self.temp) > np.random.random():
                    flipped_array[random_x,random_y] += 1
                else:
                    #Flip back to original value
                    ising_lattice[random_x,random_y] = -1*ising_lattice[random_x,random_y]
           
            average_flips = np.sum(flipped_array) / (self.size)**2
            
        return ising_lattice
        
    def get_average_energy_per_site(self,lattice,x_site,y_site):
        """ Calculate average energy per site

        Returns:
            average energy per site
        """
        energy_array = []
        spin_array = np.array(self.pairs(lattice,x_site,y_site))
        #For loop for inputting -1 for same direction and 1 for opposite direction
        for i in range(0,4):
            neighbor = spin_array[i]
            direction = neighbor + lattice[x_site,y_site]
            # Here I put 0.1 but its really just 0 but since these are all down in computers numbers are never exact
            # So i just picked a small enough number.
            if direction < 0.1 and direction > -0.1:
                energy_site = 1
            else:
                energy_site = -1
            energy_array.append(energy_site)
        energy_array = np.array(energy_array)
        average_energy = np.sum(energy_array) / self.size
        return average_energy
        
    def get_average_spin_per_site(self,lattice,x_site,y_site):
        """ Calculate average spin per site

        Returns:
            average spin per site
        """
        spin_array = self.pairs(lattice,x_site,y_site)
        average_spin = np.sum(spin_array) / self.size
        return average_spin

    def get_average_energy_and_spin(self):
        for i in range(self.size):
            for j in range(self.size):
                energy = self.get_average_energy_per_site(self.generated_lattice,i,j)
                spin = self.get_average_spin_per_site(self.generated_lattice,i,j)
                self.energy_site[i,j] = energy
                self.spin_site[i,j] = spin

    def display(self, title=''):
        """Plot the square lattice with up or down spins in different colours

        Args:
           title : of plot (default '')
        """
        self.generated_lattice = self.equilibrium()
        d = 70
        plt.figure(dpi=d)
        plt.imshow(self.generated_lattice,vmin=-1, vmax=1,)
        plt.set_cmap("Wistia")
        
    def __str__(self):
        """Prints temp, energy and spin per site"""
        self.get_average_energy_and_spin()
        _outstr = "Temperature: " + str(self.temp) + "\n"
        _outstr += "Average Energy per site: " + "\n" + str(self.energy_site) + "\n"
        _outstr+= "Average Spin per site: " + "\n" +str(self.spin_site)
        return _outstr
    
    def cool(self, temp_start, flips_per_site=1000, cooling_time=100.):
        step_index_list = np.linspace(0,600,1,endpoint = True)
        ising_lattice = self.generate_lattice()
        for i in step_index_list:
            self.temp = temp_start*np.exp(-i / self.temp)
        
            #Keeps track of how many flips each component is flipped
            flipped_array = np.zeros_like(ising_lattice)
            
            
            #Keeps track of number of flips
            average_flips = np.sum(flipped_array) / self.size
            while average_flips < flips_per_site:
            #Generates numbers between 0 to size of lattice for x and y
            
                random_x = np.random.randint(0, self.size )
                random_y = np.random.randint(0, self.size )
                
                #Rembering the old energy value before the flip
                old_energy = self.get_average_energy_per_site(ising_lattice,random_x,random_y)
                
                #Flip Spin
                ising_lattice[random_x,random_y] = -1*ising_lattice[random_x,random_y]
                
                #Calculate new energy
                new_energy = self.get_average_energy_per_site(ising_lattice,random_x,random_y)
                delta_energy = new_energy - old_energy
                if new_energy <= old_energy:
                    flipped_array[random_x,random_y] += 1
                else:
                    if np.exp((-1.0 * (delta_energy))/self.temp) > np.random.random():
                        flipped_array[random_x,random_y] += 1
                    else:
                        #Flip back to original value
                        ising_lattice[random_x,random_y] = -1*ising_lattice[random_x,random_y]
                   
                average_flips = np.sum(flipped_array) / (self.size)**2
        self.display("Cooling Lattice")
        return ising_lattice
           
   
   
   
       
'''
lattice = Ising2D(3,10)
lattice.display("Test")
print (lattice)
#lattice.display(title='Start')
'''

'''
lattice = Ising2D(temp=1000, size=20)
lattice.display(title='Start')

for i in range(12):
    lattice = Ising2D(temp=1000., size=20) # Initialize 
    lattice.temp = 1     # Rapid quenching to low T
    lattice.equilibrium(100) # Allow to come to equilibrium
    #print(lattice)            # Print temp, energy and spin per site
    
    # Display graphically
    #title = 'Rapid T=0.001 Spin:{:5.2f}'.format(lattice.get_average_spin_per_site())
    #title += ' Energy:{:5.2f}'.format(lattice.get_average_energy_per_site())
    lattice.display("Start")
'''

for i in range(10):
    lattice = Ising2D(temp=3., size=10)
    lattice.cool(temp_start=3., flips_per_site=500, cooling_time=100)
    print(lattice)
    
    # Display graphically
    title = 'Slow T=0.02 Spin:{:5.2f}'.format(lattice.get_average_spin_per_site())
    title += ' Energy:{:5.2f}'.format(lattice.get_average_energy_per_site())
    lattice.display(title=title)


