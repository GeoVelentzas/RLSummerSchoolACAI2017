import random
import gym
import numpy as np
import copy as cp
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import sys
from keras import optimizers
from keras import initializers
from collections import deque
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop






#### GLOBAL VARIABLES #######################################################################
SESSIONS = 200
#############################################################################################













############# DQN AGENT CLASS ###############################################################
# this agent will work only for carpole v1 envinronment
# with 4 dimentional continuous state vector and 2 actions
class deepagent:
    def __init__(self, env): 
        self.maxlen = 500 # memory size for experience replay
        self.gamma = 0.95 # gamma for r + gamma*Q
        self.mem = [] # initialization of memory buffer
        self.epsilon = 1.0 # initial epsilon value for exploration
        self.decay = 0.98 # decay of epsilon after every experience replay
        self.lr = 0.0001 # learning rate for the weight updates
        self.batch_size = 32 # batch size for each epoch of weight updates
        self.C = 1 # after how many experience replays from mem buffer to update target
        self.model = self.make_model() # initialize model with keras

    def make_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim = 4, kernel_initializer='random_uniform',
                bias_initializer='zeros', activation='relu'))
        model.add(Dense(24, kernel_initializer='random_uniform',
                bias_initializer='zeros', activation="relu"))
        model.add(Dense(12,kernel_initializer='random_uniform',
                bias_initializer='zeros', activation="relu"))
        model.add(Dense(2,  kernel_initializer='random_uniform',
                bias_initializer='zeros', activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return model

    def take_action(self, state):
        if random.random()<self.epsilon: 
            return random.randrange(2) #return random action with prob epsilon
        else:
            return np.argmax(self.model.predict(np.array(state).reshape(1,4)))

    def update(self):
        self.epsilon = max(self.decay*self.epsilon, 0.01) # decay epsilon up to 0.1

    def expreplay(self, memory, size, target_agent):
        memlength = len(memory) # the size of memory buffer for experience replay
        iters = 5*memlength//size # sample "size" number of samples for "iters" times from mem 
        for i in range(iters):
            print("learning from experience session: %03.2f %%" %(100.0*i/(iters-1)), end='\r')
            weights = np.array(memory)[:,5].reshape(len(memory)) #
            weights = np.array(weights, dtype='float64')
            probs = weights/np.sum(weights) #probability for choosing each quadriple from mem
            idx = np.array(range(len(weights)), dtype='int')
            indices = np.random.choice(idx, size=size, replace=False, p=probs) #indices of chosen ones
            data = np.array(memory)[indices,:].tolist() #minibatch of memory to replay
            # data = np.array(random.sample(memory, size))
            SS, TT = [] , []
            for s, a, r, sp, done, td in data:
                Q = self.model.predict(np.array(s).reshape(1,4))
                Qsp = self.model.predict(np.array(sp).reshape(1,4))
                Qhat = target_agent.model.predict(np.array(sp).reshape(1,4))
                if done:
                    Q[0][a] = r
                else:
                    # Q[0][a] = r + self.gamma*np.amax(Qhat) #Q-Learning
                    Q[0][a] = r + self.gamma*Qhat[0][np.argmax(Qsp)] #Double Q-Learning
                SS.append(s)
                TT.append(Q.tolist()[0])
            self.model.fit(np.array(SS), np.array(TT), epochs=1, verbose=0)
###############################################################################################













######### TRAIN DQN AGENT #####################################################################
def traindqn(env):
    agent = deepagent(env) #create agent
    target_agent = deepagent(env) #create target agent
    # agent.model.load_weights("./cartpolev1_dqn_weights.h5")
    # target_agent.model.load_weights("./cartpolev1_dqn_weights.h5")
    done = False
    cumlist = [0]
    session = 0
    hypermem = []
    while session<SESSIONS:
        cumreward = 0
        state = env.reset()
        while True:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            cumreward += reward
            reward = reward if not done else -10 #maybe not needed since it just shifts...
            q = agent.model.predict(np.array(state).reshape(1,4))
            qsp = agent.model.predict(np.array(next_state).reshape(1,4))
            qhat = target_agent.model.predict(np.array(next_state).reshape(1,4))
            if done:
                td = abs(q[0][action] - reward)
            else:
                # td = abs(q[0][action] - reward - agent.gamma*np.amax(qhat))
                td = abs(q[0][action] - reward - agent.gamma*qhat[0][np.argmax(qsp)]) #DoubleQ
            agent.mem.append((state, action, reward, next_state, done, td))
            state = cp.deepcopy(next_state)
            if done:
                break

        cumlist.append(0.0*cumlist[-1] + 1.0*cumreward)
        print("%03.2f %%  " % (100.0*session/SESSIONS), end=''),
        print("epsilon: %03.3f " %(agent.epsilon), end=''),
        print("lr : %03.5f " %(agent.lr), end=''),
        print("memory: %03.3f " %(len(agent.mem)), end=''),
        print("reward: %03.3f " %(cumlist[-1]), end='\n')

        if len(agent.mem)>=agent.maxlen:
            # agent.expreplay(agent.mem, agent.batch_size, target_agent)
            hypermem.append(agent.mem)
            if len(hypermem)>10:
                hypermem.pop(random.randint(0,len(hypermem)-2)) #not the last session...
            ses = 1
            for hyp in np.random.permutation(range(len(hypermem))).tolist():
                print("")
                print("running session: {}/{}".format(ses, len(hypermem)))
                agent.expreplay(hypermem[hyp], agent.batch_size, target_agent)
                target_agent.model.set_weights(agent.model.get_weights()) #update target
                ses+=1
            agent.mem = [] #flush memory
            if session%agent.C == 0:
                target_agent.model.set_weights(agent.model.get_weights()) #update target
            agent.update()
            session+=1
    agent.model.save_weights("./weights.h5") #uncomment to save weights
###############################################################################################













######### TEST DQN AGENT ######################################################################
def testdqn(env, trials):
    agent = deepagent(env)
    agent.model.load_weights("./cartpolev1_dqn_weights.h5")
    for test in range(trials):
        agent.epsilon = 0.000
        cumreward = 0
        state = env.reset()
        while True:
            env.render()
            state = np.array(state).reshape(1, 4)
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            cumreward += reward
            if done:
                break
            state = next_state
        print("cumreward: ", cumreward)
        input()
###############################################################################################













###############################################################################################
if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    # C = traindqn(env)
    # plt.plot(C)
    # plt.show()

    testdqn(env, 5)
###############################################################################################