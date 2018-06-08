import numpy as np
from scipy.ndimage.interpolation import shift
import pandas as pd
import os
from itertools import product

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
    def __init__(self, game=randomwalk(), num_sets=100, trainset_size=10):
        self.game = game
        self.num_sets = num_sets
        self.trainset_size = trainset_size
        self.dataset = dict()
    def generate_data(self, ):
        for i in range(self.num_sets):
            self.dataset[i] = dict()
            for j in range(self.trainset_size):
                self.game.simulate()
                self.dataset[i][j] = self.game.state_sequence

        return None

    def save_data(self, data_dir="data"):
        for i in range(self.num_sets):
            for j in range(self.trainset_size):
                #self.game.simulate()
                directory = os.path.join(data_dir, "trainset_{}".format(i))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                #df = pd.DataFrame(self.game.state_sequence)
                df = pd.DataFrame(self.dataset[i][j])
                df.to_csv(os.path.join(directory, "sequence_{}.csv".format(j)), header=False, index=False)
                #np.savetxt(os.path.join(directory, "sequence_{}.csv".format(j)), self.game.state_sequence, delimiter=",")

        return None
        

class tdLearner_from_simulator(object):
    def __init__(self, game=randomwalk(), alpha=0.5, decay=0.9995, lamb=0.5, init_w=None):
        self.game = game
        self.alpha = alpha
        self.lamb = lamb
        self.decay = decay
        if init_w is None:
            self.init_w = np.array([0]+[1./(game.num_states-1)]*(game.num_states-2)+[1])
        else:
            self.init_w = np.array([0]+[init_w]*(game.num_states-2)+[1])
        self.w = self.init_w.copy()

    def learn(self, maxepisode=3000, epsilon=1e-15, verbose=False):
        T = 0
        w0 = self.init_w.copy()
        delta = []
        while True: #loops through all games
            #print "new game:"
            self.game.simulate() #generate a sequence
            e = np.array([0] * (self.game.num_states))
            P0 = np.dot(w0, self.game.state_sequence[0])
            for i in range(1, self.game.state_sequence.shape[0]): #loop through game steps in 1 sequence
                P1 = np.dot(w0, self.game.state_sequence[i])
                e = (self.game.state_sequence[i-1] + e)
                delta_w = (self.decay**T)*self.alpha*(P1-P0)*e
                #delta_w[0] = 0
                #delta_w[-1] = 0
                P0 = P1
                e = e * self.lamb
                w0 += delta_w
            #print w0
            #print self.w
            delta.append(np.abs(w0 - self.w))
            #delta.append(w0 - self.w)
            if len(delta) > 20: delta.pop(0)
            #w += delta_w
            T += 1
            if T > maxepisode:
                if verbose: print "not converged in {} episode".format(T)
                break
            elif len(delta) >= 20 and np.all(np.array(delta).mean(axis=0) < epsilon):
                if verbose: print "converged in {} episode".format(T)
                break
            #print np.abs(w0 - self.w)
            self.w = w0.copy()
        
        self.findRMSE()
        return None

    def findRMSE(self, ):
        self.rmse = np.sqrt((((self.w - np.array([0, 1./6, 1./3, 1./2, 2./3, 5./6, 1.]))[1:-1])**2).mean(axis=0))
        return None




class tdLearner_from_data(object):

    def __init__(self, alpha=0.5, lamb=0.5, init_w=None, num_states=7, num_sets=100, trainset_size=10):
        self.alpha = alpha
        self.lamb = lamb
        self.data = dict()
        self.num_sets = num_sets
        self.trainset_size = trainset_size
        if init_w is None:
            self.init_w = np.array([0.]+[1./(num_states-1)]*(num_states-2)+[1.])
        else:
            self.init_w = np.array([0.]+[init_w]*(num_states-2)+[1.])
        self.w_list = []
        self.rmse = 0
        self.std = 0
        self.w = self.init_w.copy()
        self.rmse_list = []
        self.std_list = []


    def read_data(self, data_dir="data"):
        for i in range(self.num_sets):
            self.data[i] = dict()
            for j in range(self.trainset_size):
                directory = os.path.join(data_dir, "trainset_{}".format(i), "sequence_{}.csv".format(j))
                self.data[i][j] = pd.read_csv(directory, header=None)
        #print self.data[0][0].values

        return None

    def learn_all_data_repeat1(self, maxepisode=30000, epsilon=1e-15, verbose=False): #data is the np array of simulated game sequence

        #w update after each game step
        T = 0
        w0 =self.init_w.copy()
        delta = []
        while True:
            for i, j in product(range(self.num_sets), range(self.trainset_size)): #loop through all trainsets X all sequences in each train set
                
                game_sequence = self.data[i][j].values
                e = np.array([0] * (game_sequence.shape[1]))
                P0 = np.dot(w0, game_sequence[0])
                delta_w = 0
                for k in range(1, game_sequence.shape[0]): #loop through all steps in 1 game sequence
                    P1 = np.dot(w0, game_sequence[k])
                    e = (game_sequence[k-1] + e)
                    delta_w += self.alpha*(P1-P0)*e
                    #delta_w[0] = 0
                    #delta_w[-1] = 0
                    P0 = P1
                    e = e * self.lamb
                w0 += delta_w

                delta.append(np.abs(delta_w))
                
                #delta.append(np.abs(w0 - self.w))
                #delta.append(w0 - self.w)
                if len(delta) > 20: delta.pop(0)
                T += 1
                if T > maxepisode:
                    if verbose: print "not converged in {} episode".format(T)
                    break
                elif len(delta) >= 20 and np.all(np.array(delta).mean(axis=0) < epsilon):
                    #print np.array(delta)
                    if verbose: print "converged in {} episode".format(T)
                    break
                    pass
                self.w = w0.copy()
            else:
                continue
            
            
            break
        self.findRMSE()

        return None

    def learn_all_data_repeat2(self, maxepisode=30000, epsilon=1e-15, verbose=False): #data is the np array of simulated game sequence

        #w update after each game step
        T = 0
        w0 =self.init_w.copy()
        delta = []
        while True:

            for i, j in product(range(self.num_sets), range(self.trainset_size)): #loop through all trainsets X all sequences in each train set
                
                game_sequence = self.data[i][j].values
                e = np.array([0] * (game_sequence.shape[1]))
                P0 = np.dot(w0, game_sequence[0])
                delta_w = 0
                for k in range(1, game_sequence.shape[0]): #loop through all steps in 1 game sequence
                    P1 = np.dot(w0, game_sequence[k])
                    e = (game_sequence[k-1] + e)
                    delta_w = self.alpha*(P1-P0)*e
                    #delta_w[0] = 0
                    #delta_w[-1] = 0
                    P0 = P1
                    e = e * self.lamb

                    w0 += delta_w


                #delta.append(np.abs(delta_w))
                #print np.abs(w0 - self.w)
                delta.append(np.abs(w0 - self.w))

                if len(delta) > 20: delta.pop(0)
                T += 1
                if T > maxepisode:
                    if verbose: print "not converged in {} episode".format(T)
                    #print np.array(delta)
                    break
                elif len(delta) >= 20 and np.all(np.array(delta).mean(axis=0) < epsilon):
                    #print np.array(delta)
                    if verbose: print "converged in {} episode".format(T)
                    #print np.array(delta)
                    break
                    pass
                self.w = w0.copy()

            else:
                continue
            break
            
        self.findRMSE()

        return None

    def learn_all_data_repeat(self, maxepisode=30000, epsilon=1e-15, verbose=False): #data is the np array of simulated game sequence

        #w update after each game step
        T = 0
        w0 =self.init_w.copy()
        delta = []
        while True:

            for i, j in product(range(self.num_sets), range(self.trainset_size)): #loop through all trainsets X all sequences in each train set
                
                game_sequence = self.data[i][j].values
                e = np.array([0] * (game_sequence.shape[1]))
                P0 = np.dot(w0, game_sequence[0])
                delta_w = 0
                for k in range(1, game_sequence.shape[0]): #loop through all steps in 1 game sequence
                    P1 = np.dot(w0, game_sequence[k])
                    e = (game_sequence[k-1] + e)
                    delta_w = self.alpha*(P1-P0)*e
                    #delta_w[0] = 0
                    #delta_w[-1] = 0
                    P0 = P1
                    e = e * self.lamb

                    w0 += delta_w


            #delta.append(np.abs(delta_w))
            #print np.abs(w0 - self.w)
            delta.append(np.abs(w0 - self.w))

            if len(delta) > 20: delta.pop(0)
            T += 1
            if T > maxepisode:
                if verbose: print "not converged in {} episode".format(T)
                #print np.array(delta)
                break
            elif len(delta) >= 20 and np.all(np.array(delta).mean(axis=0) < epsilon):
                if verbose: print "converged in {} episode".format(T)
                break
                pass
            self.w = w0.copy()


        self.findRMSE()

        return None

    def learn_one_trainset_repeat(self, dataset_num=0, maxepisode=30000, epsilon=1e-15, verbose=False): 
        #delta_w accumulates through one iteration of a trainset
        #w update at the end of each iteration of a trainset
        if verbose: print "current trainset_{}".format(dataset_num)
        T = 0
        w = self.init_w.copy()
        delta = []
        while True:
            delta_w = 0
            for j in range(self.trainset_size): #loop through all sequences in 1 trainset
                game_sequence = self.data[dataset_num][j].values
                e = np.array([0] * (game_sequence.shape[1]))
                P0 = np.dot(w, game_sequence[0])
                for k in range(1, game_sequence.shape[0]): #loop through all steps 1 game sequence
                    P1 = np.dot(w, game_sequence[k])
                    e = (game_sequence[k-1] + e)
                    #print delta_w
                    #print self.alpha
                    #print P0
                    #print P1
                    #print e
                    delta_w += self.alpha*(P1-P0)*e
                    #delta_w[0] = 0
                    #delta_w[-1] = 0
                    P0 = P1
                    e = e * self.lamb
                    
            delta.append(np.abs(delta_w))
            if len(delta) > 20: delta.pop(0)
            #update w
            w += delta_w
            T += 1
            if T > maxepisode:
                if verbose: print "not converged in {} episode".format(T)
                break
            elif len(delta) >= 20 and np.all(np.array(delta).mean(axis=0) < epsilon):
                #print np.array(delta)
                if verbose: print "converged in {} episode".format(T)
                break
                pass
        if verbose: print "w of current trainset is:\n", w
        self.w = w.copy()
        self.findRMSE()

    def learn_one_trainset_repeat_allsets(self, maxepisode=30000, epsilon=1e-15, verbose=False): #data is the np array of simulated game sequence
        for i in range(self.num_sets):

            self.learn_one_trainset_repeat(dataset_num=i, maxepisode=maxepisode, epsilon=epsilon, verbose=verbose)
            self.w_list.append(self.w)
            self.rmse_list.append(self.rmse)
        self.rmse = np.mean(self.rmse_list)
        self.std = np.std(self.rmse_list)

        #self.findMeanRMSE()

    def learn_one_trainset(self, dataset_num=0, verbose=False):
        #delta_w accumulates through one iteration of a sequence
        #w update at the end of one iteration of a sequence
        if verbose: print "running dataset_{}".format(dataset_num)
        w = self.init_w.copy()
        for j in range(self.trainset_size): #loop through all sequences in 1 trainset
            game_sequence = self.data[dataset_num][j].values
            e = np.array([0] * (game_sequence.shape[1]))
            P0 = np.dot(w, game_sequence[0])
            delta_w = 0
            for k in range(1, game_sequence.shape[0]): #loop through all steps 1 game sequence
                P1 = np.dot(w, game_sequence[k])
                e = (game_sequence[k-1] + e)
                #print delta_w
                #print self.alpha
                #print P0
                #print P1
                #print e
                delta_w += self.alpha*(P1-P0)*e
                #delta_w[0] = 0
                #delta_w[-1] = 0
                P0 = P1
                e = e * self.lamb
            w += delta_w
            if verbose: print "w = {}".format(w) 
        self.w = w.copy()
        self.findRMSE()
        if verbose: print "rmse = {}".format(self.rmse, self.w)
        #if verbose: print "w = {}".format(self.w) 



    def learn_one_trainset_allsets(self, verbose=False):
        for i in range(self.num_sets):
            self.learn_one_trainset(dataset_num=i, verbose=verbose)
            self.w_list.append(self.w)
            self.rmse_list.append(self.rmse)
        self.rmse = np.mean(self.rmse_list)
        self.std = np.std(self.rmse_list)
        if verbose: print "alpha = {}, lamb = {}: rmse = {}".format(self.alpha, self.lamb, self.rmse)
        #self.findMeanRMSE
        #print self.rmse
        return None

    def findRMSE(self, ):
        self.rmse = np.sqrt((((self.w - np.array([0, 1./6, 1./3, 1./2, 2./3, 5./6, 1.]))[1:-1])**2).mean(axis=0))
        return None


    #def findMeanRMSE(self, ):
    #    self.rmse = np.sqrt((((np.array(self.w_list) - np.array([0, 1./6, 1./3, 1./2, 2./3, 5./6, 1.]))[:,1:-1])**2).mean(axis=1)).mean(axis=0)
    #    self.std = np.sqrt((((np.array(self.w_list) - np.array([0, 1./6, 1./3, 1./2, 2./3, 5./6, 1.]))[:,1:-1])**2).mean(axis=1)).std(axis=0)
    #    return None
                
class gridSearch(object):
    def __init__(self, learner=tdLearner_from_data, lamb=0.2):
        self.learners = []
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            self.learners.append(learner(alpha=alpha, lamb=lamb, init_w=0.0))

    def learn():
        return None







if __name__ == '__main__':

    print "run tests from run_p1.py"
    #test1
    #run_tdSimulator(alpha=0.01, lamb=0.0, init_w=0.5, maxepisode=100000, epsilon=1e-15)

    #test2 
    #run_tdAllData(alpha=0.001, lamb=0.0, init_w=0.5, maxepisode=100000, epsilon=1e-15)

    #test2
    #run_tdTrainsetRepeat(alpha=0.01, lamb=0.0, init_w=0.5, dataset_num=3, maxepisode=100000, epsilon=1e-15)


    #test3
    #run_experiment_1(alpha=0.01, lamb=0.0, init_w=0.5, maxepisode=30000, epsilon=1e-15)



    