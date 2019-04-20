import numpy as np
import tensorflow as tf
import csv

LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
BATCH_SIZE = 32

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, save_model=False, save_graph=False):
        self.MEMORY_CAPACITY = 2000
        self.learn_step_counter = 0
        self.replace_target_iter = 100
        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.save_model = save_model
        self.save_graph = save_graph
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True, name='heart_a')
            a_ = self._build_a(self.S_, scope='target', trainable=False, name='heart_a_')
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True, name='heart_q')
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False, name='heart_q_')
	
        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        tf.add_to_collection("new_ctrain",self.ctrain)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        tf.add_to_collection("new_atrain",self.atrain)

        self.sess.run(tf.global_variables_initializer())
        if self.save_graph:
            tf.summary.FileWriter("logs_DDPG/", self.sess.graph)
        self.saver = tf.train.Saver()

    def choose_action(self, s):
        #return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        return self.sess.run(self.a, {self.S: s})

    def learn(self):
        # soft target replacement
        #if self.learn_step_counter % self.replace_target_iter == 0:
            #self.sess.run(self.soft_replace)
            #print('paras changed.')

        self.sess.run(self.soft_replace)
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

    def _build_a(self, s, scope, trainable, name):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name=name)

    def _build_c(self, s, a, scope, trainable,name):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2 = tf.layers.dense(net, 1, trainable=trainable)
            return tf.multiply(net2, 1, name=name)  # Q(s,a)
            
    def save_sweatheart(self):
        if self.save_model :
            self.saver.save(self.sess, './ddpg_model_save_dir/DDPGPicassoModel')

            foo = self.memory
            with open('file'+'_2.csv', 'wb') as abc:
                np.savetxt(abc, foo, delimiter=",")
            print ("write over")

           


