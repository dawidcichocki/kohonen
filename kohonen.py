import numpy as np
import math
import matplotlib.pyplot as plt

class kohonen(object):
    
    def __init__(self, input_lenght, size_x = 25, size_y = 10,  learning_rate = 0.1 , neighbour_radious = 4, neighbour_function_type = 1):
        self.size_x = size_x
        self.size_y = size_y
        self.learning_rate = learning_rate
        self.neighbour_radious = neighbour_radious
        self.neighbour_function_type = neighbour_function_type
        self.input_lenght = input_lenght
        # initializes net with size x,y and random initial values from range [0.25,0.5)
        self.neurons = ((np.random.random((size_x, size_y, input_lenght)) + 3) /4) -0.5
    
        self.__distribution = self.gaussian_distribution()
        
        self.eta = self.learning_rate
        self.r = self.neighbour_radious
        
        self.u_matrix = None
        
    def gaussian_distribution(self, mean = 0, variance = 1):
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
        if self.neighbour_function_type:
            d = self.metric(xi,yi,xw,yw)
            if d <= r:
                return self.__distribution[math.floor((d/r)*49)]
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
        self.eta -= self.eta/10
        self.r -= self.r/2
      
    def reference_neurons(self, positive, negative):
        return self.winner(positive), self.winner(negative)
    
    #needs optimization 
    def u_matrix_calculation(self):
        u_matrix = np.zeros((2*self.size_x -1, 2*self.seize_y - 1))
        for x in range(self.size_x - 1):
            for y in range(self.size_y - 1 ):
                u_matrix[2*x, 2*y] = self.neurons[x,y]
                u_matrix[2*x +1, 2*y] = self.euclidean(self.neurons[x,y], self.neurons[x+1,y])
                u_matrix[2*x, 2*y +1] = self.euclidean(self.neurons[x,y], self.neurons[x,y+1])
                u_matrix[2*x +1, 2*y +1] = 0,5*(u_matrix[2*x +1, 2*y] + u_matrix[2*x, 2*y +1])/np.sqrt(2)
        u_matrix[2*self.size_x - 2, 2*self.size_y - 2] = self.neurons[self.size_x - 1, self.size_y - 1]
        self.u_matrix = u_matrix
        
    def get_clusters(self):
        #get watershed markers with opencv
        #get boundries
        #set clusters within boundries
        #return clusters
        pass             
        
    
    def heat_map(self, vector):
        z = self.weight_metric(vector)
        plt.imshow(z, cmap='hot', interpolation='nearest')
        plt.show()
     
        
    def reset(self):
        self.r = self.neighbour_radious
        self.eta = self.learning_rate