import sys
import os
import numpy as np
from proxy_env import RLBot
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model, Sequential
from keras.layers import Dense, Activation
from keras.utils.vis_utils import plot_model

def test():
    #load model
    print 'Loading trained model'
    model_dir = '/home/daniel/IA/v-rep/src/modelos_salvos/model_proxy_conf2_small_mazze_4actions_150_500.hdf5'
    model = load_model( model_dir )
    model_name = os.path.basename(model_dir)

    #Initializing robot
    env = RLBot()

     # Set learning parameters
    y = .99
    e = 0.1
    num_epochs = 1
    num_steps = 600
    # create lists to contain total rewards and steps per episode
    stepList = []
    rewardList = []
    lossList = []
    SPEED = 0.7
    for i in range(num_epochs):
        # Reset environment and get first new observation
        env.reset()
        state, reward = env.step([SPEED, SPEED])
        state = state['proxy_sensor'].reshape((1, -1))
        reward = reward['proxy_sensor']
        Q = model.predict(state)
        action = Q.argmax()
        rAll = 0
        done = False
        loss = 0
    #Defining num_steps
        for j in range(num_steps):
                print("Epoch {} | Step {} | Action: {} | Reward: {}".format(i, j, action, reward))
                # Choose an action by greedily (with e chance of random action)
                # from the Q-network
                Q = model.predict(state)
                action = Q.argmax()
                # Get new state and reward from environment
                speed = np.zeros(2)
                # Q -> forward, backwards, left, right
                if action == 0:
                    speed[0] = SPEED
                    speed[1] = SPEED
                if action == 1:
                    speed[0] = -SPEED
                    speed[1] = -SPEED
                if action == 2:
                    speed[0] = 0
                    speed[1] = SPEED
                if action == 3:
                    speed[0] = SPEED
                    speed[1] = 0
                if action == 4:
                    speed[0] = 0
                    speed[1] = 0
                
                state_, reward_ = env.step(speed)
                state_ = state_['proxy_sensor'].reshape((1, -1))
                reward_ = reward_['proxy_sensor']
                # Obtain the Q' values by feeding the new state through our network
                Q_ = model.predict(state_)
                # Obtain maxQ' and set our target value for chosen action.
                maxQ_ = np.max(Q_)
                targetQ = Q
                targetQ[0, action] = reward + y * maxQ_
                
                if done is True:
                    break
        # Reduce chance of random action as we train the model.
        stepList.append(j)
        rewardList.append(rAll)
        lossList.append(loss)
        print("Episode: " + str(i))
        print("Loss: " + str(loss))
        print("e: " + str(e))
        print("Reward: " + str(rAll))
        pickle.dump({'stepList': stepList, 'rewardList': rewardList, 'lossList': lossList},
                    open(model_name +'.p', "wb"))
        if done is True:
                break

if __name__ == '__main__':
    try:
        test()
    except KeyboardInterrupt:
        print('Exiting.')
