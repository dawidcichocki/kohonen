import numpy as np
import math
import matplotlib.pyplot as plt

class kohonen(object):
    
    def __init__(self, input_lenght, size_x = 25, size_y = 10,  learning_rate = 0.1 , neighbour_radious = 4, neighbour_function = 1):
        self.size_x = size_x
        self.size_y = size_y
        self.learning_rate = learning_rate
        self.neighbour_radious = neighbour_radious
        self.neighbour_function = neighbour_function
        self.input_lenght = input_lenght
        # initializes net with size x,y and random initial values from range [0.25,0.5)
        self.neurons = ((np.random.random((size_x, size_y, input_lenght)) + 3) /4) -0.5
    
        self.__distribuation = self.gaussian_distribuation()
        
        self.eta = self.learning_rate
        self.r = self.neighbour_radious
        
    def gaussian_distribuation(self, mean = 0, variance = 1):
        sigma = math.sqrt(variance)
        x=np.linspace(mean + 3 * sigma, 100)
        return x
        
    def weight_metric(self,ivector):
        #euclidian distance
        z = np.sum((self.neurons - ivector)**2, axis=-1)
        return np.sqrt(z)
    
    
    def metric(self, xi, yi, xw, yw):
        #euclidian distance
        return math.sqrt((xw-xi)**2 + (yw-yi)**2)
    
    def n_fun(self, xi, yi, xw, yw, r):
        #gaussian function
        if self.neighbour_function:
            d = self.metric(xi,yi,xw,yw)
            if d <= r:
                return self.__distribuation[math.floor((d/r)*49)]
            else:
                return 0
        
        #binary function
        else:
            return self.metric(xi, yi, xw, yw) >= r
    #todo    
    def n_matrix(self, xw, yw, r):
        nm = np.zeros((self.size_x, self.size_y))
        mx,nx,my,ny = self.boundries(xw, yw, r)
        for x in range(mx,nx):
            for y in range(my,ny):
                nm[x][y]=self.n_fun(x, y, xw, yw, r)
        return nm
                
              
    def winner(self, ivector):
        index_min = np.argmin(self.weight_metric(ivector), axis=1)
        return index_min[0], index_min[1]

    
    def boundries(self, x, y, r):
        return (math.floor(max(x - r, 0)), math.floor(min(x + r, self.size_x)), math.floor(max(y - r, 0)), math.floor(min(y + r, self.size_y)))
    
    def learn(self, input_vector):
        eta = self.eta
        r = self.r
        for ivector in input_vector:
            x, y = self.winner(ivector)
            self.neurons += eta * self.n_matrix(x, y, r)[:, :, np.newaxis] * (self.neurons - ivector)
       self.decay()
        
    #todo: exp time decay
    def decay(self):
        self.eta -= self.eta/300
        self.r -= self.r/20
        
    def reference_neurons(self, positive, negative):
        return self.winner(positive), self.winner(negative)
    
    def heat_map(self, vector):
        z = self.weight_metric(vector)
        plt.imshow(z, cmap='hot', interpolation='nearest')
        plt.show()
     
        
    def reset(self):
        self.r = self.neighbour_radious
        self.eta = self.learning_rate
