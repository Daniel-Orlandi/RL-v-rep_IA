import sys
import os
import numpy as np
from proxy_env import RLBot
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model, Sequential
from keras.layers import Dense, Activation
from keras.utils.vis_utils import plot_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



        
def train():
    #Initializing model
    print 'Building Model.'
    model_dir = '/home/daniel/IA/rl-car/src/proxy/model_3sensor.hdf5'
    model_name = os.path.basename(model_dir)
    model_name = model_name[0:13]
    
    try:
        model = load_model(model_dir)
        
        print 'Saved model found.'
    
    except:
        model = Sequential()
        model.add(Dense(units=15, input_dim=3, activation='relu'))
        model.add(Dense(units=10, activation='relu'))
        model.add(Dense(units=5, activation='relu'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy')
        
        print 'No model found, creating new.'
        print "Model Created"
        
        plot_model(model, to_file=model_name +'.png', show_shapes=True, show_layer_names=True)
        print 'Model save to file:', model_name +'.png'
        
            
    #Initializing environment
    env = RLBot()

    # Set learning parameters
    y = .99
    e = 0.2
    num_epochs = 100
    num_steps = 100
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
        
        if i <= 60 and i>30:
            num_steps = 300

        elif i > 60:
            num_steps = 500

        # The Q-Network
        for j in range(num_steps):
            print("Epoch {} | Step {} | Action: {} | Reward: {}".format(i, j, action, reward))
            # Choose an action by greedily (with e chance of random action)
            # from the Q-network
            Q = model.predict(state)
            action = Q.argmax()
            # random action
            if np.random.rand(1) < e:
                action = np.random.randint(5)
                print("e = {}. Choosing Random Action: {}".format(e, action))
            # Get new state and reward from environment
            speed = np.zeros(2)
            # Q -> forward, backwards, left, right
            if action == 0:
                speed[0] = SPEED
                speed[1] = SPEED
            if action == 1:
                speed[0] = 0
                speed[1] = 0
            if action == 2:
                speed[0] = 0
                speed[1] = SPEED
            if action == 3:
                speed[0] = SPEED
                speed[1] = 0
            if action == 4:
                speed[0] = -SPEED
                speed[1] = -SPEED
            

            state_, reward_ = env.step(speed)
            state_ = state_['proxy_sensor'].reshape((1, -1))
            reward_ = reward_['proxy_sensor']
            # Obtain the Q' values by feeding the new state through our network
            Q_ = model.predict(state_)
            # Obtain maxQ' and set our target value for chosen action.
            maxQ_ = np.max(Q_)
            targetQ = Q
            targetQ[0, action] = reward + y * maxQ_
            # Train our network using target and predicted Q values
            loss += model.train_on_batch(state, targetQ)
            rAll += reward
            state = state_
            reward = reward_
            if done is True:
                break
        # Reduce chance of random action as we train the model.
        if num_epoch <= 50:
            e -= 0.001
        else:
            e -= 0.002 #0.01
            
        
        stepList.append(j)
        rewardList.append(rAll)
        lossList.append(loss)
        print("Episode: " + str(i))
        print("Loss: " + str(loss))
        print("e: " + str(e))
        print("Reward: " + str(rAll))
        pickle.dump({'stepList': stepList, 'rewardList': rewardList, 'lossList': lossList},
                    open(model_name + str(num_epochs) + '_' + str(num_steps) +'.p', "wb"))
        model.save(model_name + str(num_epochs) + '_' + str(num_steps) +'.hdf5')
        if done is True:
                break
    
    print("Average loss: " + str(sum(lossList) / num_epochs))
    print("Average number of steps: " + str(sum(stepList) / num_epochs))
    print("Average reward: " + str(sum(rewardList) / num_epochs))

    plt.plot(rewardList)
    plt.plot(stepList)
    plt.plot(lossList)
    

    

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        print('Exiting.')
