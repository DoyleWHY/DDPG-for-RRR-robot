from DDPG_Picasso_Env import Picasso_Sim
import tensorflow as tf
import numpy as np
import math
import time
import tkinter
import tkinter.messagebox

LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
BATCH_SIZE = 32

class DDPG(object):
	def __init__(self, a_dim, s_dim, a_bound, save_model=False,):
	    self.MEMORY_CAPACITY = 2000
	    self.learn_step_counter = 0
	    self.replace_target_iter = 100
	    self.tmp = np.loadtxt("file_2.csv", delimiter=",")
	    self.memory=np.array(self.tmp)
	    self.pointer = 0
	    self.save_model = save_model
	    self.sess = tf.Session()

	    self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,

	    new_saver = tf.train.import_meta_graph('./ddpg_model_save_dir/DDPGPicassoModel.meta')
	    new_saver.restore(sess, tf.train.latest_checkpoint('./ddpg_model_save_dir'))
	    graph = tf.get_default_graph()
	    #tf.summary.FileWriter("logs_Picasso/", sess.graph)
	    
	    self.S = graph.get_tensor_by_name("s:0")
	    self.S_ = graph.get_tensor_by_name("s_:0")
	    self.R = graph.get_tensor_by_name("r:0")
	    
	    self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
	    self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
	    self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
	    self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
	    self.soft_replace1 = [tf.assign(t, (1 - TAU) * t + TAU * e)
							 for t, e in zip(self.at_params, self.ae_params)]
	    self.soft_replace2 = [tf.assign(t, (1 - TAU) * t + TAU * e)
							for t, e in zip(self.ct_params, self.ce_params)]
							
	    self.a = graph.get_tensor_by_name("Actor/eval/heart_a:0")
	    
	    self.ctrain=tf.get_collection("new_ctrain")[0]
	    self.atrain=tf.get_collection("new_atrain")[0]
	    
	    self.sess.run(tf.global_variables_initializer())

	    self.saver = tf.train.Saver()

	def choose_action(self, s):
	    return self.sess.run(self.a, {self.S: s})

	def learn(self):
	    self.sess.run(self.soft_replace1)
	    self.sess.run(self.soft_replace2)
	    indices = np.random.choice(self.MEMORY_CAPACITY, size=BATCH_SIZE)
	    bt = self.memory[indices, :]
	    bs = bt[:, :self.s_dim]
	    ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
	    br = bt[:, -self.s_dim - 1: -self.s_dim]
	    bs_ = bt[:, -self.s_dim:]

	    self.sess.run(self.atrain, {self.S: bs})
	    self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        #self.learn_step_counter += 1

	def store_transition(self, s, a, r, s_):
	    transition = np.hstack((s, a, [[r]], s_))
	    index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
	    self.memory[index, :] = transition
	    self.pointer += 1
        #print(transition)
	   

RENDER = True

MAX_EPISODES = 200
MAX_EP_STEPS = 500

def run_picasso_demo():
	var = 0.01
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
		    var *= .9995    # decay the action randomness
		    ddpg.learn()

		    s = s_
		    ep_reward += r
		    print('Cycles:',i,'  Steps:',steps)
		    if done or steps >= MAX_EP_STEPS:
			    if done:
				    tkinter.messagebox.showinfo('Thank God','Model works!')
				    #time.sleep(2)
				    #env.target_position[0,0] = input("Input new x：");
				    #env.target_position[0,1] = input("Input new y：");
				    #env.target_position[0,2] = input("Input new z：");	
				    #env.target_position[0,0] = random.uniform(-0.85, 0.85)
				    #env.target_position[0,1] = random.uniform(0.15, 0.85)
				    #env.target_position[0,2] = random.uniform(0.15, 0.85)
			    print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, ' Steps:',steps)
			    break
	print('Running time: ', time.time() - t1)
	
	
with tf.Session() as sess:
	env = Picasso_Sim() 
	s_dim = 6
	a_dim = 3
	a_bound = np.array([0.03*math.pi, 0.03*math.pi, 0.03*math.pi])
	
	ddpg = DDPG(a_dim, s_dim, a_bound, save_model=False)
	run_picasso_demo()


