from p1 import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


#test to run
def run_tdSimulator(alpha=0.01, decay=0.9995, lamb=0.0, init_w=0.5, maxepisode=100000, epsilon=1e-15):
    tdSimulator = tdLearner_from_simulator(alpha=alpha, decay=decay, lamb=lamb, init_w=init_w)
    print "initial w is:/n", tdSimulator.w
    tdSimulator.learn(maxepisode=maxepisode, epsilon=epsilon)

    print tdSimulator.w
    print tdSimulator.rmse
    #print tdSimulator.std

    return None

def run_tdAllData(data_dir="data", alpha=0.01, lamb=0.0, init_w=0.5, maxepisode=100000, epsilon=1e-15):
    tdData = tdLearner_from_data(alpha=alpha, lamb=lamb, init_w=init_w)
    tdData.read_data(data_dir)
    tdData.learn_all_data_repeat(maxepisode=maxepisode, epsilon=epsilon)
    print np.array(tdData.w)
    print tdData.rmse
    #print tdData.std
    return None

def run_tdTrainsetRepeat(data_dir="data", alpha=0.01, lamb=0.0, init_w=0.5, dataset_num=0, maxepisode=30000, epsilon=1e-15):
    tdTrainset = tdLearner_from_data(alpha=alpha, lamb=lamb, init_w=init_w)
    tdTrainset.read_data(data_dir)
    tdTrainset.learn_one_trainset_repeat(dataset_num=dataset_num, maxepisode=maxepisode, epsilon=epsilon)
    print np.array(tdTrainset.w)
    print tdTrainset.rmse
    #print tdTrainset.std
    return None


def run_tdTrainsetRepeat_allSets(data_dir="data", alpha=0.01, lamb=0.0, init_w=0.5, maxepisode=30000, epsilon=1e-15, verbose=False):
    experiment = tdLearner_from_data(alpha=alpha, lamb=lamb, init_w=init_w)
    experiment.read_data(data_dir)
    experiment.learn_one_trainset_repeat_allsets(maxepisode=maxepisode, epsilon=epsilon, verbose=verbose)

    #print experiment.rmse
    #print experiment.std
    #print np.array(experiment.w_list).mean(axis=0)
    return experiment.rmse, experiment.std

def run_experiment_1(data_dir="data", results_dir="results", lamb_range=np.arange(0., 1.05, 0.1)):

    rmse_list = []
    std_list = []
    for lamb in lamb_range:
        print "lamb = {}".format(lamb)
        rmse, std = run_tdTrainsetRepeat_allSets(data_dir=data_dir, alpha=0.01, lamb=lamb)
        rmse_list.append(rmse)
        std_list.append(std)
        print "rmse = {}\nstd = {}".format(rmse, std)
    print np.array(rmse_list)
    print np.array(std_list)
    result = np.array([rmse_list, std_list]).T
    result = pd.DataFrame(result, index=lamb_range, columns=["Error", "STD"])
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result.to_csv(os.path.join(results_dir, "experiment1.csv"), header=True, index=True)


    return None


def plot_experiment_1(results_dir, name):
    csv_file = os.path.join(results_dir, "{}.csv".format(name))

    fig_file = os.path.join(results_dir, "{}.png".format(name))
    df = pd.read_csv(csv_file, header=0, index_col=0)
    print df
    #plt.errorbar(df.index, 'Error', yerr='STD', data=df)
    #plt.show()
    ax = df[["Error"]].plot(style=["ro-"])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylabel("Error")
    ax.set_xlabel(r"$\lambda$")
    
    plt.savefig(fig_file)

def run_tdTrainset(data_dir="data", alpha=0.01, dataset_num=0, lamb=0.0, init_w=0.5, verbose=False):
    experiment = tdLearner_from_data(alpha=alpha, lamb=lamb, init_w=init_w)
    experiment.read_data(data_dir)
    experiment.learn_one_trainset(dataset_num=dataset_num, verbose=verbose)

    #print experiment.rmse
    #print experiment.std
    #print np.array(experiment.w_list).mean(axis=0)
    return experiment.rmse, experiment.std

def run_tdTrainset_allSets(data_dir="data", alpha=0.01, lamb=0.0, init_w=0.5, verbose=False):
    experiment = tdLearner_from_data(alpha=alpha, lamb=lamb, init_w=init_w)
    experiment.read_data(data_dir)
    experiment.learn_one_trainset_allsets(verbose=verbose)

    #print experiment.rmse
    #print experiment.std
    #print np.array(experiment.w_list).mean(axis=0)
    return experiment.rmse, experiment.std
    


def run_experiment_2(data_dir="data", results_dir="results", alpha_range = np.arange(0., 0.65, 0.05), lamb_range=[0., 0.3, 0.8, 1.], init_w=0.5, verbose=False):
    result = dict()
    for lamb in lamb_range:
        result[r"$\lambda$={}".format(lamb)] = []
        for alpha in alpha_range:
            print "lamb = {}, alpha = {}".format(lamb, alpha)
            experiment = tdLearner_from_data(alpha=alpha, lamb=lamb, init_w=init_w)
            experiment.read_data(data_dir)
            experiment.learn_one_trainset_allsets(verbose=verbose)
            #print np.array(experiment.w_list).mean(axis=0)
            #print experiment.rmse, experiment.std
            result[r"$\lambda$={}".format(lamb)].append(experiment.rmse)

    result = pd.DataFrame(result)
    result.index = alpha_range
    print result
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result.to_csv(os.path.join(results_dir, "experiment2.csv"), header=True, index=True)


def plot_experiment_2(results_dir, name):
    csv_file = os.path.join(results_dir, "{}.csv".format(name))

    fig_file = os.path.join(results_dir, "{}.png".format(name))
    fig_file1 = os.path.join(results_dir, "{}_1.png".format(name))
    df = pd.read_csv(csv_file, header=0, index_col=0)
    print df

    ax = df.plot(style=["ro-", "bs-", "g^-", "yv-"])
    ax.set_xlim(-0.04, 0.65)
    ax.set_ylim(0, 1.4)
    ax.set_ylabel("Error")
    ax.set_xlabel(r"$\alpha$")
    #plt.show()
    plt.savefig(fig_file)

    ax = df.plot(style=["ro-", "bs-", "g^-", "yv-"])
    ax.set_xlim(-0.04, 0.65)
    ax.set_ylabel("Error")
    ax.set_xlabel(r"$\alpha$")
    #plt.show()
    plt.savefig(fig_file1)

def run_experiment_3(data_dir="data", results_dir="results", alpha_range = np.arange(0., 0.65, 0.05), lamb_range=np.arange(0., 1.05, 0.1), init_w=0.5, verbose=False):
    result = dict()
    for lamb in lamb_range:
        result[lamb] = []
        for alpha in alpha_range:
            print "lamb = {}, alpha = {}".format(lamb, alpha)
            experiment = tdLearner_from_data(alpha=alpha, lamb=lamb, init_w=init_w)
            experiment.read_data(data_dir)
            experiment.learn_one_trainset_allsets(verbose=verbose)
            #print np.array(experiment.w_list).mean(axis=0)
            #print experiment.rmse, experiment.std
            result[lamb].append(experiment.rmse)

    result = pd.DataFrame(result)
    result.index = alpha_range
    print result
    result2 = result.T
    print result2
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    result2.to_csv(os.path.join(results_dir, "experiment3_1.csv"), header=True, index=True)
    result2=result2.min(axis=1)
    result2.columns = ["Error"]
    result2.to_csv(os.path.join(results_dir, "experiment3_2.csv"), header=True, index=True)

def plot_experiment_3(results_dir, name):
    csv_file = os.path.join(results_dir, "{}.csv".format(name))

    fig_file = os.path.join(results_dir, "{}.png".format(name))
    df = pd.read_csv(csv_file, header=0, index_col=0)
    print df
    df.columns = ["Error"]
    print df
    #plt.errorbar(df.index, 'Error', yerr='STD', data=df)
    #plt.show()
    ax = df.plot(style=["ro-"])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylabel("Error")
    ax.set_xlabel(r"$\lambda$")
    
    plt.savefig(fig_file)


if __name__ == '__main__':
    
    data_dir = "data_0"
    results_dir = "results_0"

    #generate data from simulation
    game = data_gen(game=randomwalk(num_states=7), num_sets=100, trainset_size=10)
    game.generate_data()
    game.save_data(data_dir=data_dir)

    #test1
    #run_tdSimulator(alpha=0.1, decay=0.999, lamb=0.0, init_w=0.5, maxepisode=100000, epsilon=1e-15)

    #test2 
    #run_tdAllData(data_dir="data", alpha=0.01, lamb=0.2, init_w=0.5, maxepisode=100, epsilon=1e-5)

    #test3
    #run_tdTrainsetRepeat(data_dir="data", alpha=0.55, lamb=0.0, init_w=0.5, dataset_num=78, maxepisode=100000, epsilon=1e-15)

    #test
    #rmse, std = run_tdTrainset(data_dir="data", alpha=0.55, dataset_num=78, lamb=0.0, init_w=0.5, verbose=True)
    #print "rmse = {}".format(rmse)

    #test
    #rmse, std = run_tdTrainset_allSets(data_dir="data", alpha=0.55, lamb=0.0, init_w=0.5, verbose=True)

    #test4
    #rmse, std = run_tdTrainsetRepeat_allSets(data_dir="data", alpha=0.01, lamb=0.0, init_w=0.5, maxepisode=30000, epsilon=1e-15, verbose=True)
    #print "rmse = {}\nstd = {}".format(rmse, std)


    #experiment 1
    #run_experiment_1(data_dir=data_dir, results_dir=results_dir)
    #plot_experiment_1(results_dir=results_dir, name="experiment1")

    #experiment 2
    run_experiment_2(data_dir=data_dir, results_dir=results_dir)
    plot_experiment_2(results_dir=results_dir, name="experiment2")

    #experiment 3
    #run_experiment_3(data_dir=data_dir, results_dir=results_dir)
    #plot_experiment_3(results_dir=results_dir, name="experiment3_2")
