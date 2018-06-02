import numpy as np
from scipy.ndimage.interpolation import shift
import pandas as pd
import os

class randomwalk(object): #1d random walk with odd number of states
    def __init__(self, num_states=7):
        assert num_states % 2 == 1, "num_states needs to be an odd integer"
        self.num_states = num_states
        self.s = np.array([0]*(num_states//2)+[1]+[0]*(num_states//2))
        self.score = 0
        self.game_on = True
        self.state_sequence = self.s.reshape(1,-1)
        self.score_sequence = []


    def move(self, ): #move 1 step

        if self.s[0] == 1: 
            self.score = 0
            self.game_on = False
        elif self.s[-1] == 1:
            self.score = 1
            self.game_on = False
        else:
            self.s = shift(self.s, shift=np.random.choice([-1,1]),order=0,cval=0)
            self.state_sequence = np.append(self.state_sequence, [self.s], axis=0)

        return None

    def simulate(self, ):
        self.s = np.array([0,0,0,1,0,0,0])
        self.game_on=True
        self.state_sequence = self.s.reshape(1,-1)
        self.score_sequence = []
        while self.game_on:
            self.move()
        return None

class data_gen(object): #generate data from simulations of the ransom walk
    def __init__(self, game=randomwalk(), num_sets=100, num_sequences=10):
        self.game = game
        self.num_sets = num_sets
        self.num_sequences = num_sequences
        self.dataset = dict()
    def generate_data(self, num_sets=100, num_sequences=10):
        for i in range(num_sets):
            self.dataset[i] = dict()
            for j in range(num_sequences):
                self.game.simulate()
                self.dataset[i][j] = self.game.state_sequence

        return None

    def save_data(self, num_sets=100, num_sequences=10):
        for i in range(num_sets):
            for j in range(num_sequences):
                #self.game.simulate()
                directory = os.path.join("data", "trainset_{}".format(i))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                #df = pd.DataFrame(self.game.state_sequence)
                df = pd.DataFrame(self.dataset[i][j])
                df.to_csv(os.path.join(directory, "sequence_{}.csv".format(j)), header=False, index=False)
                #np.savetxt(os.path.join(directory, "sequence_{}.csv".format(j)), self.game.state_sequence, delimiter=",")

        return None
        

class tdLearner_from_simulator(object):
    def __init__(self, game=randomwalk, alpha=0.5, lamb=0.5, init_w=None):
        self.game = game()
        self.alpha = alpha
        self.lamb = lamb
        if init_w is None:
            self.w = np.array([0]+[1./(game.num_states-1)]*(game.num_states-2)+[1])
        else:
            self.w = np.array([0]+[init_w]*(game.num_states-2)+[1])

    def learn(self, maxepisode=3000):
        T=0
        delta = []
        while True: #loops through all games
            #print "new game:"
            self.game.simulate()
            e = np.array([0] * (self.game.num_states))
            P0 = np.dot(self.w, self.game.state_sequence[0])
            w0 = self.w.copy()
            for i in range(1, self.game.state_sequence.shape[0]): #loop through game steps
                P1 = np.dot(w0, self.game.state_sequence[i])
                e = (self.game.state_sequence[i-1] + e)
                delta_w = self.alpha*(P1-P0)*e
                delta_w[0] = 0
                delta_w[-1] = 0
                P0 = P1
                e = e * self.lamb
                w0 += delta_w
            #print w0
            #print self.w
            delta.append(np.abs(w0 - self.w))
            if len(delta) > 20: delta.pop(0)
            if T > maxepisode:
                print T
                break
            #if len(delta) >= 20 and np.all(np.array(delta).mean(axis=0) < 0.005):
            #    print T
            #    break
            #print np.abs(w0 - self.w)
            self.w = w0.copy()

            T += 1
            return None

class tdLearner_from_data(object):

    def __init__(self, alpha=0.5, lamb=0.5, init_w=None, num_states=7, num_sets=100, num_sequences=10):
        self.alpha = alpha
        self.lamb = lamb
        self.data = dict()
        self.num_sets = num_sets
        self.num_sequences = num_sequences
        if init_w is None:
            self.init_w = np.array([0.]+[1./(num_states-1)]*(num_states-2)+[1.])
        else:
            self.init_w = np.array([0.]+[init_w]*(num_states-2)+[1.])
        self.w = []
        self.rmse = 0
        self.std = 0

    def read_data(self, ):
        for i in range(self.num_sets):
            self.data[i] = dict()
            for j in range(self.num_sequences):
                directory = os.path.join("data", "trainset_{}".format(i), "sequence_{}.csv".format(j))
                self.data[i][j] = pd.read_csv(directory, header=None)
        print self.data[0][0].values

        return None

    def learn(self, ): #data is the np array of simulated game sequence
        for i in range(self.num_sets):
            w1 = self.init_w.copy()
            for j in range(self.num_sequences):
                game_sequence = self.data[i][j].values
                e = np.array([0] * (game_sequence.shape[1]))
                P0 = np.dot(self.init_w, game_sequence[0])
                w0 = w1.copy()
                for k in range(1, game_sequence.shape[0]): #loop through game steps
                    P1 = np.dot(w0, game_sequence[k])
                    e = (game_sequence[k-1] + e)
                    delta_w = self.alpha*(P1-P0)*e
                    delta_w[0] = 0
                    delta_w[-1] = 0
                    P0 = P1
                    e = e * self.lamb
                    w0 += delta_w
                w1 = w0.copy()
            self.w.append(w1)

    def findRMSE(self, ):
        self.rmse = ((((np.array(self.w) - np.array([0, 1./6, 1./3, 1./2, 2./3, 5./6, 1.]))[:,1:-1])**2).mean(axis=1)).mean(axis=0)
        self.std = ((((np.array(self.w) - np.array([0, 1./6, 1./3, 1./2, 2./3, 5./6, 1.]))[:,1:-1])**2).mean(axis=1)).std(axis=0)
        return None
                
class gridSearch(object):
    def __init__(self, learner=tdLearner_from_data, lamb=0.2):
        self.learners = []
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            self.learners.append(learner(alpha=alpha, lamb=lamb, init_w=0.0))

    def learn():
        return None




if __name__ == '__main__':
    #a = tdLearner_from_simulator(alpha=0.4, lamb=0.2, init_w=0.0)
    #print a.w
    #a.learn(maxepisode=1000)
    
    #print a.game.score
    #print a.game.state_sequence
    #print a.w

    #b = data_gen()
    #b.generate_data()
    #b.save_data()
    #print b.dataset[20][5]

    c = tdLearner_from_data(alpha=0.4, lamb=0.0, init_w=0.0)
    c.read_data()
    c.learn()
    c.findRMSE()
    print c.rmse
    print c.std



    