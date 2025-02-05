"""
This file contains the agents' information.
"""

import numpy as np
import random
import math


np.random.seed(None)
class Agent():
    """
    Agent is the one who makes decisions, based on the current state.
    One agent is just for one machine.
    It should load the pre-trained nn model
    """
    def __init__(self, px, py, pgx, pgy,radius, max_speed,  kinematic,  time_constraint =1000):
        self.px = px
        self.py = py
        self.H = 50
        self.vx = 0
        self.vy = 0
        self.radius = radius
        self.pgx = pgx
        self.pgy = pgy
        self.maxspeed =max_speed
        self.theta = (math.atan2(self.pgy-self.py, self.pgx-self.px) + 2*math.pi) % (2*math.pi)
        self.kinematic = kinematic
        self.kinematic_constraint = math.pi/3
        # self.kinematic_constraint = math.pi
        self.done = False
        self.vo_record_length = 2
        self.vo_record=[]
        self.close_range = 6  # How many times of maxspeed that I think is too close.
        self.actions = self.action_space()
        self.time_constraint = time_constraint
        self.time_left = time_constraint
        # self.time = 0
        self.EE = 0
        self.charge = 120000

    def get_current_v(self):
        return [self.vx, self.vy]

    def get_current_p(self):
        return [self.px, self.py]

    def get_pref_angle(self):
        return (math.atan2(self.pgy-self.py, self.pgx-self.px) + 2*math.pi) % (2*math.pi)

    def get_dis_to_goal(self):
        return math.sqrt(  (self.pgy-self.py)**2+ (self.pgx-self.px )**2)

    def get_full_state(self):
        return [self.px, self.py, self.H, self.vx, self.vy, self.radius, self.pgx, self.pgy, self.maxspeed, self.theta, self.EE,self.charge]

    def get_observable_state(self):
        return [self.px, self.py, self.H, self.vx, self.vy, self.radius]

    def action_space(self, kinematic = True):
        """
        Action space v2
        """
        if kinematic:
            actions = []
            just_rotate = np.linspace(start=-self.kinematic_constraint, stop=self.kinematic_constraint, num=7, endpoint=True)
            # just_speed = self.maxspeed*[(5 - i) / 5 * self.maxspeed for i in range(5)]
            just_speed = [self.maxspeed]
            for rosp in just_speed:
                for roan in just_rotate:
                    actions.append([rosp, roan])    # rosp速度，roan转向角
            actions.append([0,self.kinematic_constraint]) #悬停
            actions.append([1,0])   #充电
            actions.append([2,0])   #upload
            return actions
        else:
            actions=[]
            just_rotate = np.linspace(start=0, stop=2*math.pi, num=4,endpoint=False)
            for roan in just_rotate:
                actions.append([self.maxspeed,roan])
            # actions.append([0,0])
            return actions




