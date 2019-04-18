"""
Build Picasso environment
"""

from gym import spaces
import numpy as np
import time
import math

action_high = np.array([0.03*math.pi, 0.03*math.pi, 0.03*math.pi])
observation_high = np.array([1., 1., 2., 1., 1., 2.])

class Picasso_Sim(object):
    def __init__(self):
        self.largeA1 = 1
        self.largeA2 = 1
        self.largeA3 = 1
        self.FRESH_TIME = 0.01         #fresh time for one move
        self.angle = np.array([[math.pi/2, math.pi/2, math.pi/2]])
        self.target_position = np.array([[0.5, 0.6, 0.7]])
        self.S_absolute = np.array([[0.,0.,0.]])
        self.S_relavite = np.array([[0.,0.,0.]])
        self.n_features = 11
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-observation_high, high=observation_high, dtype=np.float32)
        #self._build_environment()
        
    def _build_environment(self):
        env_list1 = ['-']*(self.n_features-1) +['O'] + ['-']*(self.n_features-1) # '-----T' our initial environment
        env_list2 = ['-']*(self.n_features-1) +['O'] + ['-']*(self.n_features-1)
        env_list3 = ['-']*(self.n_features-1) +['O'] + ['-']*(self.n_features-1)*2
        interaction1 = ''.join(env_list1)
        interaction2 = ''.join(env_list2)
        interaction3 = ''.join(env_list3)
        print('\r{}'.format(interaction1), end='\n')
        print(format(interaction2), end='\n')
        print(format(interaction3), end='\n')

    def reset(self):
        # return observation
	    self.step_counter = 0
	    self.angle = np.array([[math.pi/2, math.pi/2, math.pi/2]])
	    self.S_absolute = np.array([[0.,0.,0.]])
	    self.S_relavite = np.array([[0.,0.,0.]])
	    return np.array([[0, 1, 0.5, self.target_position[0,0]-0, self.target_position[0,1]-1, self.target_position[0,2]-0.5]])
	    
    def render(self, S, episode, step_counter):
		# This is how environment be updated
	    env_list1 = ['-']*(20+5) +['5'] + ['-']*(15) # '--------T' our environment
	    env_list2 = ['-']*(20+6) +['6'] + ['-']*(14)
	    env_list3 = ['-']*(20+7) +['7'] + ['-']*(23)
	    
	    if abs(S[0,0]) <= 1 and abs(S[0,1]) <= 1 and (S[0,2] <= 2 or S[0,2] >= -1):
		    disPositionX_A = int(S[0,0]*10) + 20
		    disPositionY_A = int(S[0,1]*10) + 20
		    disPositionZ_A = int(S[0,2]*10) + 20
		    env_list1[disPositionX_A] = 'x'
		    env_list2[disPositionY_A] = 'y'
		    env_list3[disPositionZ_A] = 'z'
			
		    interaction1 = ''.join(env_list1)
		    interaction2 = ''.join(env_list2)
		    interaction3 = ''.join(env_list3)
		    print('\r{}'.format(interaction1), end='\n')
		    print(format(interaction2), end='\n')
		    print(format(interaction3), end='\n')
		    time.sleep(self.FRESH_TIME)	    
		    
    def step(self, S, action):
        # This is how agent will interact with the environment 
	    self.angle += action
	    if self.angle[0, 0] <= 0:
		    self.largeA1 = -self.angle[0, 0]
		    self.angle[0, 0] = 0.0
	    elif self.angle[0, 0] >= math.pi:
		    self.largeA1 = self.angle[0, 0] - math.pi
		    self.angle[0, 0] = math.pi
	    else:
		    self.largeA1 = 0
			
	    if self.angle[0, 1] <= 0:
		    self.largeA2 = -self.angle[0, 1]
		    self.angle[0, 1] = 0.0
	    elif self.angle[0, 1] >= math.pi:
		    self.largeA2 = self.angle[0, 1] - math.pi
		    self.angle[0, 1] = math.pi
	    else:
		    self.largeA2 = 0
			
	    if self.angle[0, 2] <= 0:
		    self.largeA3 = -self.angle[0, 2]
		    self.angle[0, 2] = 0.0
	    elif self.angle[0, 2] >= math.pi:
		    self.largeA3 = self.angle[0, 2] - math.pi
		    self.angle[0, 2] = math.pi
	    else:
		    self.largeA3 = 0
			
	    #print('angle:',self.angle)
	    theta1 = self.angle[0, 0]
	    theta2 = self.angle[0, 1]
	    theta3 = self.angle[0, 2]
	    
	    self.S_absolute[0, 0] = math.cos(theta1)*math.sin(theta2) + (math.cos(theta1)*math.cos(theta2)*math.sin(theta3))/2 + (math.cos(theta1)*math.cos(theta3)*math.sin(theta2))/2
	    self.S_absolute[0, 1] = math.sin(theta1)*math.sin(theta2) + (math.cos(theta2)*math.sin(theta1)*math.sin(theta3))/2 + (math.cos(theta3)*math.sin(theta1)*math.sin(theta2))/2
	    self.S_absolute[0, 2] = math.cos(theta2) + (math.cos(theta2)*math.cos(theta3))/2 - (math.sin(theta2)*math.sin(theta3))/2 + 1
	    
	    self.S_relavite[0, 0] = self.target_position[0, 0] - self.S_absolute[0, 0]
	    self.S_relavite[0, 1] = self.target_position[0, 1] - self.S_absolute[0, 1]
	    self.S_relavite[0, 2] = self.target_position[0, 2] - self.S_absolute[0, 2]   

	    distance_target = (self.S_absolute[0, 0]-self.target_position[0, 0])**2 + (self.S_absolute[0, 1]-self.target_position[0, 1])**2 + (self.S_absolute[0, 2]-self.target_position[0, 2])**2
	    print('distance:',distance_target)
	    if distance_target <= 0.03:
		    reward = 50
		    done = True
	    elif distance_target > 0.03 and distance_target <= 0.06:
		    reward = 40
		    done = False
	    elif distance_target > 0.06 and distance_target <= 0.09:
		    reward = 30
		    done = False
	    elif distance_target > 0.09 and distance_target <= 0.12:
		    reward = 20
		    done = False
	    elif distance_target > 0.12 and distance_target <= 0.15:
		    reward = 10
		    done = False
	    elif distance_target > 0.15 and distance_target <= 0.20:
		    reward = 5
		    done = False
	    else:
		    #reward = 0
		    reward = -(self.largeA1+self.largeA2+self.largeA3)*10
		    #reward = -distance_target*100
		    done = False
	    
	    """
	    if abs(self.S_relavite[0, 0]) <= 0.1 and abs(self.S_relavite[0, 1]) <= 0.1 and abs(self.S_relavite[0, 2]) <= 0.1:
		    reward = 30
		    done = True
	    elif abs(self.S_relavite[0, 0]) <= 0.2 and abs(self.S_relavite[0, 1]) <= 0.2 and abs(self.S_relavite[0, 2]) <= 0.2 and abs(self.S_relavite[0, 0]) > 0.1 and abs(self.S_relavite[0, 1]) > 0.1 and abs(self.S_relavite[0, 2]) > 0.1:
		    reward = 10
	    else:
		    reward = 0
		    #reward = -self.largeA1*self.largeA2*self.largeA3
		    #reward = -(abs(self.S_relavite[0, 0]*10)**2+abs(self.S_relavite[0, 1]*10)**2+abs(self.S_relavite[0, 2]*10)**2)
		    done = False
		    """
		    
	    S_ = np.hstack((self.S_absolute, self.S_relavite))    
	    return S_, reward, done




