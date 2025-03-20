"""
This is the environment. The agents interact with this environment, execute actions, and get rewards.
"""
import numpy as np
import random
import scipy.io as sio
import math

np.random.seed(5)
class Env():
    """
    The environment gets the inputs of the agents, process the action, and give reward.
    It also check with the episode is ended or not.
    """
    def __init__(self):
        self.bound_r = 50
        self.landing_size = 10     #
        self.landing_width = 20
        self.counter = 0
        # self.penalty = -1 # Penalty when collision happens
        # self.penalty_sinr = -1
        # self.bonus = 2    # Bonus for arriving goal.
        self.penalty_obs, self.penalty_bdy, self.penalty_uncharge = -0.2, -0.2,  -500 #default penalty_other was -10
        self.bonus_node, self.bonus_upload_goal, self.bonus_closer_n = 2, 5, 0 #gpu12,20，此12,20
        self.other_dp = 10 #default 0.2
        self.UAV_H = 50
        self.a = 0

        # self.nodes_r = 10  # minimum distance for communication, default = 10
        self.nodesPt = 10**(-3)  # transmit power of IoT nodes
        self.N = 10**(-6)    #noise
        self.Tsinr = 0.37 #  #default 0.37/0.3 for 10m, 0.39 for 5m, jammer: 3.5
        self.average_aoi = []

        ## define the obstacles or non-fly zones
        self.obstacles = [[-23, -23, 6], [20, 20, 6]]  # [x,y,r]
        # self.obstacles = []
        self.B = 1e6
        self.num = 0
        self.charge = [0, -0, 6]

    def reset_env(self):
        """
        Generate the starts and finals according to the landing areas.
        Generate the starts and goals for other agents.
        Generate the nodes.
        """
        self.reward_aoi = 0
        self.radius = 0.5
        self.maxspeed = 5
        self.average_aoi = []

        self.uav_buffer = []
        self.uav_buffer_real_data = []
        self.upload_data =[]
        self.num_of_charge = 0
        self.total_success_load = 0
        self.total_charge_can_beused = 120000
        # start position
        sx = np.random.choice(a= np.arange(start=-self.bound_r, stop =-self.bound_r+self.landing_size+1, step=10 ))
        sy = np.random.choice(a= np.arange(start=self.bound_r-self.landing_width, stop =self.bound_r, step=10 ))
        # sx = np.random.choice(a=np.arange(start=-self.bound_r, stop=-self.bound_r + self.landing_size + 1, step=10))
        # sy = np.random.choice(a=np.arange(start=- self.landing_width/2, stop=self.landing_width/2, step=10))

        # g = np.random.choice([2,3])
        # landing position
        self.gx = 0
        self.gy = 0
        # information of IoT nodes
        # self.num_nodes = np.random.choice([5,6,7,8,9,10])
        self.num_nodes =50
        self.num_of_upload = 0
        # self.num_pkg_each = np.random.choice([1,2,3]) # it's suggested to be 1
        # if num_nodes = 5, num_pkg_each=2, then nodes_packages = [2,2,2,2,2]
           ####

        # Define a cluster of nodes -- method 1
        choice = np.arange(start=- self.bound_r/5*4, stop=self.bound_r/5*4, step=0.5)
        self.nodesX = np.random.choice(choice, size=self.num_nodes,replace=False)
        self.nodesY = np.random.choice(choice, size=self.num_nodes,replace=False)
        self.nodes_packages = self.simulate_data_generation()


        return [sx,sy,self.gx,self.gy,self.radius, self.maxspeed]

    def generate_data_packet(self,node_id, replay_buffer, time, position):

        packet = {
            'node_id': node_id,
            'data': replay_buffer,
            'timestamp': time,

        }
        return packet

    def simulate_data_generation(self):
        data_packets = []
        for i in range(self.num_nodes):
            x = self.nodesX[i]
            y = self.nodesY[i]
            position = [x, y]
            node_replay = []
            time = []
            packet = self.generate_data_packet(i, node_replay, time, position)
            data_packets.append(packet)
        return data_packets

    def update_nodes(self,time):
        self.a += 1
        b = self.a % 20 == 0
        for i in range(self.num_nodes):
            if b:
                a = round(random.uniform(-0.5, 0.5), 2)
                b = round(random.uniform(-0.5, 0.5), 2)
                self.nodesX[i] += a
                self.nodesY[i] += b
                if self.nodesX[i] < -45 or self.nodesX[i] > 45:
                    self.nodesY[i] -= a
                if self.nodesY[i] < -45 or self.nodesY[i] > 45:
                    self.nodesY[i] -= b



    def observe_nodes(self,time):
        '''
        The typical agent get information of nodes
        '''
        self.random_process_nodes(time)
        st_nodes = []
        for i in range(self.num_nodes):
            # st_nodes.append([self.nodesX[i], self.nodesY[i], self.nodes_packages[i]['data'],
            #                  self.nodes_packages[i]['timestamp'], self.nodes_packages[i]['expired']])
            st_nodes.append([self.nodesX[i], self.nodesY[i], self.nodes_packages[i]['data'],
                             self.nodes_packages[i]['timestamp']])

        return st_nodes

    def random_process_nodes(self,time):
        num_nodes_to_process = 1
        if num_nodes_to_process > 0:
            for i in random.sample(range(50), num_nodes_to_process):
                self.add_packet(self.nodes_packages[i], time)
        self.update_nodes(time)


    def step(self,action, agent, time, sort_nodes_index, sort_node_state):
        '''
        Agents execute their actions
        '''
        reward_charge = 0
        sp,ag = action

        theta = (agent.theta + ag + 2*math.pi) % (2*math.pi)
        vx,vy = sp*math.cos(theta), sp*math.sin(theta)
        if sp == 5:
            ec = 215
            agent.charge -= ec
        elif sp == 0:
            ec = 222
            agent.charge -= ec
        elif sp== 1:
            vx, vy = 0, 0
            dis_uav_charge = math.sqrt((agent.px-self.charge[0])**2+(agent.py-self.charge[1])**2) <= 5
            if dis_uav_charge:
                if agent.charge <= 20000:
                    self.num_of_charge += 1
                    reward_charge = 5
                    ec = 0
                    agent.charge = 120000
                else:
                    self.num_of_charge += 1
                    ec = 0
                    reward_charge = -40
                    agent.charge = 120000

            else:
                ec = 222
                reward_charge = -5
                agent.charge -= ec
        else:
            vx, vy = 0, 0
            ec = 222
            agent.charge -= ec

        px_, py_,vx_,vy_, reward, done, sinr, Trans_rate = self.reward(sp,agent.px, agent.py, vx, vy, agent.pgx, agent.pgy, time, agent.charge, sort_nodes_index,sort_node_state)
        reward += reward_charge
        agent.px, agent.py, agent.vx,agent.vy,agent.theta,agent.done = px_,py_, vx_, vy_,theta,done
        agent.time_left -= 1

        return agent, reward, done,ec, Trans_rate


    def reward(self, sp, px, py, vx, vy, pgx, pgy, time,charge,nodes_index,sort_node_state):
        '''
        Environment returns reward after executing one action.
        '''
        reward_obs, reward_bdy, reward_cha, reward_other, reward_node, reward_dis, reward_goal, done = 0, 0, 0, 0, 0, 0, 0, 0
        reward_age = 0
        Trans_rate = 0
        check_obs = self.check_obstacles(px + vx, py + vy)
        check_bdy = self.check_boundary(px + vx, py + vy)
        check_battery = self.check_battery(charge)
        if check_obs or check_bdy or check_battery:
            px_, py_ = px, py
            vx_, vy_ = 0.0, 0.0
            if check_obs:
                reward_obs = self.penalty_obs
            if check_bdy:
                reward_bdy = self.penalty_bdy
            if check_battery:
                reward_cha = self.penalty_uncharge

        else:
            px_, py_ = px + vx, py + vy
            vx_, vy_ = vx, vy

        # encourage the UAV to move closer to a node
        #schedule 1
        # dis_,dis_a_,dis_truth_ = self.get_dis_to_nodes(px_, py_, time)
        # n = np.argmin(dis_)

       # sp3
        dis = self.get_dis_to_nodes(px_, py_, time, nodes_index, sort_node_state)
        a = np.argmin(dis)
        n = nodes_index[a]

        # # schedule 4
        # dis = self.get_dis_to_nodes(px_, py_, time, nodes_index, sort_node_state)
        #
        # a = np.argmin(dis)
        # n = nodes_index[a]

        # schedule 5
        # dis_a_ = self.get_dis_to_nodes(px_, py_, time)
        # n = np.argmin(dis_a_)




        pxn, pyn = self.nodesX[n], self.nodesY[n]
        dis = math.sqrt((px_ - pxn) ** 2 + (py_ - pyn) ** 2)
        sinr = self.get_sinr(px_,py_,dis)
        package = self.nodes_packages[n]
        length_of_data = len(package['data'])

        if sinr >= self.Tsinr and length_of_data > 0:
            package['data'] = []
            node_times = package['timestamp']
            reward_node = self.bonus_node*length_of_data
            for i in range(length_of_data):
                self.uav_buffer.append(node_times[i])
            package['timestamp'] = []
            Trans_rate = self.B * math.log2(1 + sinr)
            real_data = Trans_rate*length_of_data
            self.uav_buffer_real_data.append(real_data)


        if charge <= 0:
            done = 5

        if time == 999:
            done = 4
        # self.reward_aoi = np.mean(self.uav_buffer)
        buffer_aoi=[]
        uav_buffer_length = len(self.uav_buffer)
        dg_ = math.sqrt((px_ - pgx) ** 2 + (py_ - pgy) ** 2)
        if uav_buffer_length == 0:
            self.reward_aoi = 0

        else:
            for i in range(uav_buffer_length):
                difference_value = time - self.uav_buffer[i]
                buffer_aoi.append(difference_value)
            self.reward_aoi = np.mean(buffer_aoi)
        if self.reward_aoi >=100:
            reward_age = -2
        if sp==2:
            if self.check_reached_goal(dg_):
            # print('uav_buffer,',uav_buffer_length)
                reward_goal = self.bonus_upload_goal*uav_buffer_length
                self.average_aoi.extend(buffer_aoi)
                self.uav_buffer = []

                self.total_success_load += uav_buffer_length
                self.num_of_upload += 1
                self.upload_data.append(sum(self.uav_buffer_real_data))
                self.uav_buffer_real_data = []
            else:
                reward_goal = -2
        reward = reward_age+reward_obs + reward_bdy + reward_goal+reward_cha + reward_node -self.reward_aoi/200

        # print('rereward_expired{},reward_obs{},reward_bdy{},reward_goal{},reward_cha{},reward_node{}'
        #       .format(self.reward_expired, reward_obs, reward_bdy, reward_goal, reward_cha, reward_node))
        # print('total reward:',reward)
        return px_,py_, vx_,vy_,reward, done, sinr, Trans_rate

    def add_packet(self, package, time):
        buffer = package['data']
        times = package['timestamp']
        buffer.append(1)
        times.append(time)
        return

    def get_dis_to_nodes(self,px,py, time, nodes_index, sort_node_state):
        dis =[] # save every distance
        dis_a = []
        dis_truth = []
        for i,idx in enumerate(nodes_index):
            packet = sort_node_state[i]
            # sp1
            # dis = (math.sqrt((px - self.nodesX[idx]) ** 2 + (py - self.nodesY[idx]) ** 2))+ (
            #         packet[7] - time) + packet[4]
            #sp2
            # dis = (math.sqrt((px - self.nodesX[idx]) ** 2 + (py - self.nodesY[idx]) ** 2)) + (
            #         packet[7] - time)
            # sp3
            # dis = (math.sqrt((px - self.nodesX[idx]) ** 2 + (py - self.nodesY[idx]) ** 2)) + packet[4]
            #sp4
            # dis = packet[7] - time
            #sp5
            dis = math.sqrt((px - self.nodesX[idx]) ** 2 + (py - self.nodesY[idx]) ** 2)
        return dis



    def get_sinr(self,px,py,dis,dis_jam=1000):
        # top = self.nodesPt * self.UAV_H * (
        #             0.8 * (dis ** 2 + self.UAV_H ** 2) ** (-1.5) + 0.2*(dis ** 2 + self.UAV_H ** 2) ** (-2.5))

        top = self.nodesPt*self.UAV_H* (dis**2 + self.UAV_H**2)**(-1.5)
        sinr = top/self.N
        # print("px,py ({},{}), top {}, jI {}, SINR {}".format(px,py,top,jI,sinr))
        return sinr

    def check_battery(self, charge):
        if charge <= 0:
            return True
        else:
            return False

    def check_boundary(self, px, py):
        """
        Check whether agent goes out of boundary
        """
        if px > self.bound_r or px < -self.bound_r or py > self.bound_r or py < -self.bound_r:
            return True
        else:
            return False

    def check_obstacles(self,px,py):
        """
        Check whether collide with obstacles
        :return: True for collision, False for otherwise
        """
        for obs in self.obstacles:
            ox,oy,r = obs
            if ox < px < ox+r and oy < py < oy+r:
                return True
        return False


    def check_reached_goal(self, dg):
        """
        Check whether agent has reached their goals.
        """
        if dg < self.maxspeed:  # radius:
            return True
        else:
            return False

# if __name__ == '__main__':
#     a=Env()
#     a.simulate_data_generation()
