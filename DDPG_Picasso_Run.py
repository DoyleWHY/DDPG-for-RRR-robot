from DDPG_Picasso_Env import Picasso_Sim
from DDPG_Picasso_Brain import DDPG
import numpy as np
import time
import math
import tkinter
import tkinter.messagebox

OUTPUT_GRAPH = False
RENDER = True

MAX_EPISODES = 200
MAX_EP_STEPS = 500

def run_picasso_demo():
	var = 0.1
	t1 = time.time()
	for i in range(MAX_EPISODES):
	    s = env.reset()
	    ep_reward = 0
	    steps = 0
	    while True:
		    steps += 1
		    if RENDER:
			    env.render(s, i, env.step_counter)

		    # Add exploration noise
		    a = ddpg.choose_action(s)
		    #print('old a:',a)
		    a = np.clip(np.random.normal(a, var), -math.pi*0.02, math.pi*0.02)    # add randomness to action selection for exploration
		    s_, r, done = env.step(s, a)
		    #print('old state:',s)
		    #print('new a:',a)
		    print('reward:',r)
		    #print('new state:',s_)

		    ddpg.store_transition(s, a, r, s_)

		    if ddpg.pointer > ddpg.MEMORY_CAPACITY:
			    var *= .9995    # decay the action randomness
			    ddpg.learn()

		    s = s_
		    ep_reward += r
		    print('Cycles:',i,'  Steps:',steps)
		    if done or steps >= MAX_EP_STEPS:
			    if done:
				    tkinter.messagebox.showinfo('Thank God','Model works!')
				    #time.sleep(2)
				    env.target_position[0,0] = input("Input new x：");
				    env.target_position[0,1] = input("Input new y：");
				    env.target_position[0,2] = input("Input new z：");	
			    print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, ' Steps:',steps)
			    break
	print('Running time: ', time.time() - t1)

if __name__ == "__main__":
    # nuggets game
    env = Picasso_Sim() 
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    
    ddpg = DDPG(a_dim, s_dim, a_bound)
       

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs_DDPG/", sess.graph)

    run_picasso_demo()
