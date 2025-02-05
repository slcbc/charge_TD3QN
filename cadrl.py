"""
The main running function.
You can define the parameters in args.
You can choose agents in the main: choose dueling DDQN.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import argparse
import numpy as np
import scipy.io as sio



from charge.D3QN_charge.environment import Env
from charge.D3QN_charge.agent import Agent
from charge.D3QN_charge.utility import Utility
from charge.D3QN_charge.replay_memory import ReplayMemory


from charge.D3QN_charge.dueling_ddqn_agent import DuelingDDQNAgent



parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target network smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.5, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--target_update_interval', type=int, default=1000, metavar='N',
                    help='Value target update per no. of updates per step (default: 1000)')
parser.add_argument('--start_to_train_steps', type=int, default=40000, metavar='N',
                    help='number of steps needed before training (default: 40000)')
parser.add_argument('--rotate_state', type=bool, default=False, metavar='G',
                    help='whether rotate the state to agent-centric coordinate')
parser.add_argument('--sinr', type=bool, default=True, metavar='G',
                    help='whether use sinr information as part of the state of nodes')
parser.add_argument('--change_node', type=bool, default=True, metavar='G',
                    help='whether to change the number of nodes')
parser.add_argument('--max_nodes', type=int, default=10, metavar='G',
                    help='random number of nodes and random slcoations')
parser.add_argument('--epsilon', type=float, default=0.6, metavar='G',
                    help='inital epsilon (default: 0.5)')   # epsilon_init
parser.add_argument('--eps_dec', type=float, default=1e-6, metavar='G',
                    help='epsilon decrement per learn (default: 1e-6)') # epsilon_decay_steps
parser.add_argument('--eps_min', type=float, default=0.1, metavar='G', # epsilon_min
                    help='epsilon minimum (default: 0.1)')
parser.add_argument('--max_obv', type=int, default=2, metavar='G',
                    help='maximum num of other agents taken into account (default: 2)')
parser.add_argument('--max_obv_region', type=float, default=10, metavar='G',
                    help='maximum observable region around each agent (default: 10)')



my_args = parser.parse_args()

np.random.seed(8)
env = Env()
util = Utility()


class Cadrl():
    def __init__(self, num_actions=10, args=my_args, alg_name= None):
        if args.rotate_state:   # False
            self.st_dim = 7
            self.sn_dim = 5 * args.max_nodes
        else:
            self.st_dim = 13
            self.sn_dim = 8 * args.max_nodes

        self.s_dim = [self.st_dim, self.sn_dim]  # 无人机状态、节点状态
        self.num_actions = num_actions
        self.args = args
        self.alg_name = alg_name

        self.main_alg = DuelingDDQNAgent(self.s_dim,  num_actions,  args)

        self.count = 0

        # ## Initial memory
        self.memory = ReplayMemory(args.replay_size, args.seed) # replay_buffer
        # self.repaly_buffer在经验池的地方改
        # ##do not rotate, random 5 locations for nodes.
        if not args.rotate_state:
            self.mean = [[3.870870377897096, -0.0005903817708431805,0,0, 57.3575567, 5.163517046344822, -0.00010821041362364053, 1000.0, 0.7055268, 0,0,0,0],
                          [[0.5572,  0.49849, 38.30513074,  3.10583315,  173.04198600466512,
            0.2091395,  0.04191, 611.5525880262135], [10.5572, 10.49849, 40.24990122,  2.75731462,  173.04198600466512,
            0.20234745,  0.04151, 611.5525880262135], [-9.4428, 10.49849, 40.04915604,  2.83014175,  173.04198600466512,
            0.2030535 ,  0.04193, 611.5525880262135], [-9.4428, -9.50151, 39.85797542,  3.39653039,  173.04198600466512,
            0.20382362,  0.04235, 611.5525880262135], [10.5572, -9.50151, 40.06156338,  3.44721946,  173.04198600466512,
            0.20307383,  0.04169, 611.5525880262135], [ 0.5572,  0.49849, 38.30513074,  3.10583315,  173.04198600466512,
            0.2091395 ,  0.04191, 611.5525880262135], [10.5572, 10.49849, 40.24990122,  2.75731462,  173.04198600466512,
            0.20234745,  0.04151, 611.5525880262135], [-9.4428, 10.49849, 40.04915604,  2.83014175,  173.04198600466512,
            0.2030535 ,  0.04193, 611.5525880262135], [-9.4428, -9.50151, 39.85797542,  3.39653039,  173.04198600466512,
            0.20382362,  0.04235, 611.5525880262135], [10.5572, -9.50151, 40.06156338,  3.44721946,  173.04198600466512,
            0.20307383,  0.04169, 611.5525880262135]]]
            self.std = [[0.9543651927631189, 3.0175402644891505,1,1, 5.06980,1.2714593813438326,0.6963553052082826, 1.0, 0.1530,1,1,1,1],
                         [[28.86316733, 28.9071579 , 14.21204168,  1.81605273,  14.659965838580296 ,
            0.07931523,  0.20038351, 69.01657811489255], [28.86316733, 28.9071579 , 16.44154561,  1.96343907,  14.659965838580296,
            0.08595445,  0.19946659, 69.01657811489255], [28.86316733, 28.9071579 , 16.25287884,  1.58227391,  14.659965838580296,
            0.08543032,  0.20042923, 69.01657811489255], [28.86316733, 28.9071579 , 16.10879155,  1.59302113,  14.659965838580296,
            0.0851147 ,  0.20138639, 69.01657811489255], [28.86316733, 28.9071579 , 16.29449322,  1.97962717,  14.659965838580296,
            0.08556607,  0.19987982, 69.01657811489255], [28.86316733, 28.9071579 , 14.21204168,  1.81605273,  14.659965838580296 ,
            0.07931523,  0.20038351, 69.01657811489255], [28.86316733, 28.9071579 , 16.44154561,  1.96343907,  14.659965838580296,
            0.08595445,  0.19946659, 69.01657811489255], [28.86316733, 28.9071579 , 16.25287884,  1.58227391,  14.659965838580296,
            0.08543032,  0.20042923, 69.01657811489255], [28.86316733, 28.9071579 , 16.10879155,  1.59302113,  14.659965838580296,
            0.0851147 ,  0.20138639, 69.01657811489255], [28.86316733, 28.9071579 , 16.29449322,  1.97962717,  14.659965838580296,
            0.08556607,  0.19987982, 69.01657811489255]]]

        else:
            self.mean = [[2.4074480330242762, 2.402802409002709, 70.28628130271905, 0.5, 5.0, 2.032027183987712, 49.49805],  [[38.97984178,  3.0531724 ,  1.99949   ,  0.20679935,  0.04191   ], [4.51250682e+01, 2.50752974e+00, 2.00006000e+00, 1.85494412e-01,
       4.20700000e-02], [40.92952279,  2.9840538 ,  1.99331   ,  0.1999856 ,  0.04201   ], [40.09748035,  3.4167529 ,  1.99593   ,  0.20294402,  0.0414    ], [4.27891463e+01, 3.75145619e+00, 1.99437000e+00, 1.93663877e-01,
       4.16300000e-02]]]
            self.std = [[2.5912178921345004, 2.5915872331566376, 28.27551627530273, 0.0, 0.0, 2.000435356067437, 28.786112210535507],  [[15.05932652,  1.63125326,  1.41341775,  0.08189483,  0.20038351], [20.57163775,  1.34393124,  1.41589548,  0.09734824,  0.20074889], [17.14946543,  1.45362257,  1.41363547,  0.08804782,  0.20061196], [16.32796219,  2.00261853,  1.41431023,  0.08566747,  0.19921355], [18.86317889,  1.97287838,  1.41361179,  0.09299235,  0.19974219]]]


    def one_trajectory(self, train, display_step=False, display_traj=True):
        '''
        Trajectory for in one whole episode
        '''
        rewards_sequences = []
        charge_sequences = []
        sx, sy, gx, gy, radius, maxspeed = env.reset_env()
        self.agent = Agent(sx, sy, gx, gy, radius, maxspeed, kinematic=True)
        trajectory = []
        node_trajectory = []

        node_state_numpak = []
        node_state_avg_time = []


        EC = 0  # 初始能量
        R = 0
        time = 0
        st = self.agent.get_full_state()
        snt = env.observe_nodes(time)
        stt = self.agent.time_left

        s_t_jn_r, sort_nodes_index, sort_node_state = util.rotate_state(env,[st, snt, stt],max_nodes=self.args.max_nodes, max_obv_region=self.args.max_obv_region,rotate=self.args.rotate_state,change_node=self.args.change_node)
        # print('1',s_t_jn_r)
        s_t_jn_s = util.standardize_state(s_t_jn_r, self.mean, self.std)
        # self.visualize_trajectory_start([sx, sy], [gx, gy], trajectory)
        while True:
            self.count += 1

            act_index = self.main_alg.choose_action(s_t_jn_s, train)
            action = self.agent.actions[act_index]
            # if self.epoch %10 == 0: print('Actor action is: ',act_index)


            self.agent, reward, done,ec, Trans_rate = env.step(action, self.agent, time, sort_nodes_index,sort_node_state)  #action is [sp,ang]l

            mask = done > 0
            R += Trans_rate
            EC += ec
            self.agent.EE = (R / EC)/300    #300

            st1 = self.agent.get_full_state()
            snt1 = env.observe_nodes(time)
            stt1 = self.agent.time_left
            reward += self.agent.EE #gpu100



            s_t1_jn_r,sort_nodes_index,sort_node_state = util.rotate_state(env,[st1, snt1, stt1],max_nodes=self.args.max_nodes, max_obv_region=self.args.max_obv_region,rotate=self.args.rotate_state,change_node=self.args.change_node)
            # print(s_t1_jn_r)
            s_t1_jn_s = util.standardize_state(s_t1_jn_r, self.mean, self.std)
            self.memory.push(s_t_jn_s, act_index, reward, s_t1_jn_s, mask)
            rewards_sequences.append(reward)
            charge_sequences.append(self.agent.charge)

            if train and self.count > self.args.start_to_train_steps:
                self.main_alg.learn(self.memory, self.args.batch_size, self.count)

            s_t_jn_s = s_t1_jn_s

            # s_t_jn_r = s_t1_jn_r

            trajectory.append([st1[0], st1[1]])


            if train:
                tt = 1000
            else:
                tt = 1000
            time += 1
            # node_trajectory.append([env.nodesX[40],env.nodesY[40]])
            # for x,y in node_trajectory:
            # print('x:{},y{},reward{},action{}'.format(self.agent.px, self.agent.py,reward,act_index))
            if display_step:
                print('step: {}'.format(time))
                self.visualize_trajectory([sx, sy], [gx, gy], trajectory)

            if done or time >= tt:  # When two agents arrive goals.
                print('Let us check whether charge:', done,
                      '. \t Packages collected:', env.total_success_load)
                if done >= 4:
                    trajectory.append([gx, gy])


                if display_traj:
                    self.visualize_trajectory([sx, sy], [gx, gy], trajectory)
                break


        total_collected = env.total_success_load
        num_of_charge = env.num_of_charge
        len_avg_aoi = len(env.average_aoi)
        if len_avg_aoi != 0:
            average_aoi=np.mean(env.average_aoi)
        else:
            average_aoi= 0
        # time_flag = (self.agent.time_left >= 0)

        # self.visualize_node_trajectory([sx, sy], [gx, gy],node_trajectory)
        # print('av', average_aoi)

        # return done, time_flag, rewards_sequences, total_collected, self.agent.EE*300,average_aoi
        # return done, time_flag, rewards_sequences, total_collected, self.agent.EE * 300, expn, average_aoi, \
        #        stt, self.agent.EE, self.agent.charge, node_state_numpak, node_state_avg_time
        return total_collected, num_of_charge, EC, self.agent.EE * 300, charge_sequences,\
               done,average_aoi,rewards_sequences, env.num_of_upload,env.upload_data

    def train(self, success_failure_path,sg,set, train, display_step=False, display_traj=False):
        """
        The training.
        """
        window = 100
        success, failure, noend,  succ_time = [], [], [], []
        success_rate, failure_rate, noend_rate = [], [], []
        succ_time_rate = []
        time_enough = []




        totoal_collected_pack,avg_collected_pack = [],[]
        totoal_num_charge,avg_num_charge=[],[]
        totoal_num_upload, avg_num_upload = [], []
        totoal_num_data, avg_num_data = [], []
        total_ec,avg_ec=[],[]
        total_ee,avg_ee=[],[]
        total_battery,avg_battery=[],[]
        total_reward, reward_arr = [], []
        total_av_aoi = []
        avg_av_aoi = []
        if train:
            # num_epoch = 20001
            num_epoch = 20000
        else:
            num_epoch = 1001
            self.main_alg.load_model(name="trained", sg = sg,set=set, suffix=".pkl")

        # total_uav_time= []
        # total_uav_energy= []
        # total_uav_charge= []
        # total_node_state_numpak,total_node_state_avg_time =[],[]

        for epoch in range(num_epoch):
            print('epoch ---------- {}'.format(epoch))
            self.epoch = epoch
            if train:
                distraj = display_traj*(epoch%20==0)
            else:
                distraj = display_traj
            # done, time_flag, rewards_sequences, pkg, energy, expkg, aoi,\
            # stt,EE,charge,node_state_numpak,node_state_avg_time = self.one_trajectory(train, display_step=display_step, display_traj=distraj)
            total_collected, num_of_charge, EC, EE, charge_sequences,\
               done,average_aoi,rewards_sequences,upload,data \
             = self.one_trajectory(train, display_step=display_step,display_traj=distraj)
               # 1表示数据包的大小
            if done== 4:
                success.append(1)
                failure.append(0)

            elif done== 5:
                success.append(0)
                failure.append(1)
                noend.append(0)
            elif done == 0:
                success.append(0)
                failure.append(0)
                noend.append(1)

            # if time_flag:
            #     tt.append(1)
            # else:
            #     tt.append(0)
            print('reward:',sum(rewards_sequences))
            total_reward.append(sum(rewards_sequences))

            totoal_collected_pack.append(total_collected)
            totoal_num_charge.append(num_of_charge)
            totoal_num_data.append(sum(data))
            total_ec.append(EC)
            total_ee.append(EE)
            totoal_num_upload.append(upload)
            total_battery.append(np.mean(charge_sequences))
            total_av_aoi.append(average_aoi)



            # total_uav_time.append(stt)
            # total_uav_energy.append(EE)
            # total_uav_charge.append(charge)

            # total_node_state_numpak.append(sum(node_state_numpak))
            # total_node_state_avg_time.append(sum(node_state_avg_time))

            if epoch > 0:
                success_rate.append(np.mean(success[-window:]))
                failure_rate.append(np.mean(failure[-window:]))
                noend_rate.append(np.mean(noend[-window:]))
                reward_arr.append(np.mean(total_reward[-window:]))

                succ_time_rate.append(np.mean(succ_time[-window:]))
                avg_collected_pack.append(np.mean(totoal_collected_pack[-window:]))


                avg_num_charge.append(np.mean(totoal_num_charge[-window:]))
                avg_num_upload.append(np.mean(totoal_num_upload[-window:]))
                # avg_num_data.append(np.mean(totoal_num_data[-window:]))
                avg_ec.append(np.mean(total_ec[-window:]))
                avg_ee.append(np.mean(total_ee[-window:]))
                avg_battery.append(np.mean(total_battery[-window:]))
                avg_av_aoi.append(np.mean(total_av_aoi[-window:]))


            if train:
                if epoch>0 and epoch % 50 ==0:
                    # sio.savemat(success_failure_path, {'success': success_rate, 'failure': failure_rate, 'notend': noend_rate,
                    #                                    'time_enough': time_enough, 'reward_avg': reward_arr, 'reward': total_reward,
                    #                                    'pkglft_avg': pkg_arr, 'pkg_left': pkg_left, 'succ_time': succ_time_rate,
                    #                                    'energy_avg': avg_energy, 'avg_ex_package': avg_ex_package,
                    #                                    'average_aoi':avg_av_aoi})
                    sio.savemat(success_failure_path,
                                {'success': success_rate, 'failure': failure_rate,
                                 'total_collected_node': avg_collected_pack, 'charge':avg_num_charge,
                                 'ec': avg_ec, 'ee': avg_ee, 'battery': avg_battery,
                                 'avg_aoi':avg_av_aoi,'reward': reward_arr})
                    self.main_alg.save_model("trained", sg, set, ".pkl")
            else:
                if epoch == num_epoch-1:

                    print('Overall: sr is %.3f, total_upload_data is %.3f,totoal_charge is %.3f,ec is %.3f,ee is %.3f,battery is %.3f,aoi is %.3f,reward is %.3f'% (np.mean(success),
                                                                                                                                                              np.mean(totoal_collected_pack),np.mean(totoal_num_charge),
                                                                                                                                                              np.mean(total_ec),np.mean(total_ee),np.mean(total_battery),
                                                                                                                                                              np.mean(total_av_aoi),np.mean(total_reward))
                          )
                    print('upload{}'.format(np.mean(totoal_num_upload)),np.mean(totoal_num_data))
                    # print('uavmean:stt{},EE{},aoi{}'.format(np.mean(total_uav_time),np.mean(total_uav_energy),np.mean(total_uav_charge)))
                    # print('uavstd:stt{},EE{},aoi{}'.format(np.std(total_uav_time), np.std(total_uav_energy), np.std(total_uav_charge)))
                    # print('nodemean:pak{},time{}'.format(np.mean(total_node_state_numpak), np.mean(total_node_state_avg_time),
                    #                        ))
                    # print('nodestd:pak{},time{}'.format(np.std(total_node_state_numpak), np.std(total_node_state_avg_time)))

    def visualize_trajectory(self, starts, goals, trajectory):
        """
        Visualize the whole trajectory in one figure.
        """
        fig, ax = plt.subplots()
        ax.set_ylim(-env.bound_r, env.bound_r)
        ax.set_xlim(-env.bound_r, env.bound_r)
        ax.grid(True)
        ax.scatter(env.nodesX, env.nodesY, s=50, c='green', marker='^', alpha=1)
        obs = env.obstacles
        for oi in obs:
            ax.add_patch(
                patches.Rectangle(xy=(oi[0], oi[1]), width=oi[2], height=oi[2], edgecolor='grey', facecolor='grey',
                                  fill=True))
        cg = env.charge
        ax.add_patch(
            patches.Rectangle(xy=(cg[0], cg[1]), width=cg[2], height=cg[2], edgecolor='orange',
                              facecolor='blue', fill=True))
        sx, sy = -env.bound_r, env.bound_r - env.landing_width
        # sx, sy = -env.bound_r, - env.landing_width/2
        ax.add_patch(
            patches.Rectangle(xy=(sx, sy), width=env.landing_width, height=env.landing_width, edgecolor='green',
                              facecolor='blue', fill=True))
        gx, gy = env.bound_r - env.landing_width, -env.bound_r
        ax.add_patch(patches.Rectangle(xy=(gx, gy), width=env.landing_width, height=env.landing_size, edgecolor='green',
                                       facecolor='green', fill=True))
        # gx1, gy1 = env.bound_r - env.landing_width, env.bound_r-env.landing_size
        # ax.add_patch(patches.Rectangle(xy=(gx1, gy1), width=env.landing_width, height=env.landing_size, edgecolor='green',
        #                                facecolor='green', fill=True))

        for (nx, ny) in zip(env.nodesX, env.nodesY):
            ax.add_patch(  #10.5 or 5.3     ##改为了5
                patches.Circle(xy=(nx, ny), radius=5.7, edgecolor='green', facecolor='grey', fill=False, linestyle=':'))

        pgx, pgy = goals
        sx, sy = starts
        ax.plot(pgx, pgy, color='navy', markersize=10, marker='X')
        ax.plot(sx, sy, color='navy', markersize=5, marker='o')
        px, py = [], []
        px.append(sx)
        py.append(sy)
        for step, traj in enumerate(trajectory):
            x, y = traj
            px.append(x)
            py.append(y)
            # ax.plot(x, y, color='navy', markersize=2, marker='.')


        ax.plot(px, py, color='navy', markersize=1, marker='o')


        # plt.ion()
        # plt.pause(2)
        # plt.close()
        plt.show()
    def visualize_node_trajectory(self, starts, goals, trajectory):
        """
        Visualize the whole trajectory in one figure.
        """
        fig, ax = plt.subplots()
        ax.set_ylim(-35, -30)
        ax.set_xlim(-37, -34)
        ax.grid(True)
        # ax.scatter(env.nodesX, env.nodesY, s=50, c='green', marker='^', alpha=1)
        # obs = env.obstacles
        # for oi in obs:
        #     ax.add_patch(
        #         patches.Rectangle(xy=(oi[0], oi[1]), width=oi[2], height=oi[2], edgecolor='grey', facecolor='grey',
        #                           fill=True))
        # sx, sy = -env.bound_r, env.bound_r - env.landing_width
        # # sx, sy = -env.bound_r, - env.landing_width/2
        # ax.add_patch(
        #     patches.Rectangle(xy=(sx, sy), width=env.landing_width, height=env.landing_width, edgecolor='green',
        #                       facecolor='blue', fill=True))
        # gx, gy = env.bound_r - env.landing_width, -env.bound_r
        # ax.add_patch(patches.Rectangle(xy=(gx, gy), width=env.landing_width, height=env.landing_size, edgecolor='green',
        #                                facecolor='green', fill=True))

        nodeX,nodeY=[],[]
        for x,y in trajectory[:5]:
            nodeX.append(x)
            # print(x)
            nodeY.append(y)
            # ax.add_patch(  #10.5 or 5.3     ##改为了5
            #     patches.Circle(xy=(x, y), radius=5, edgecolor='green', facecolor='grey', fill=False, linestyle=':'))
        ax.scatter(nodeX, nodeY, s=30, c='green', marker='o', alpha=1)
        for i in range(1, 5):
            dx = nodeX[i] - nodeX[i-1]
            dy = nodeY[i] - nodeY[i-1]
            plt.arrow(nodeX[i - 1], nodeY[i - 1], dx*0.5, dy*0.5, head_width=0.05, head_length=0.08, fc='blue', ec='blue')

        pgx, pgy = goals
        sx, sy = starts
        ax.plot(pgx, pgy, color='navy', markersize=10, marker='X')
        ax.plot(sx, sy, color='navy', markersize=5, marker='o')
        px, py = [], []
        px.append(sx)
        py.append(sy)



        ax.plot(px, py, color='navy', markersize=1, marker='o')


        # plt.ion()
        # plt.pause(2)
        # plt.close()
        plt.show()

    def visualize_step(self, starts, goals, positions):
        """
        Visualize each step in the trajectory.
        """
        fig, ax = plt.subplots()
        ax.set_ylim(-60, 60)
        ax.set_xlim(-60, 60)
        ax.grid(True)

        # ## visualize the BSs
        ax.scatter(env.BSX, env.BSY, s=20, c='green', marker='^', alpha=1)
        ax.scatter(self.UEX, self.UEY, s=2, c='yellow', marker='.', alpha=0.5)

        # ## visualize the starts and goals
        col = ['b', 'g', 'r', 'c', 'm', 'k']
        rnd = np.random.choice(col, size=5, replace=False)
        for i in range(len(positions)):
            pgx, pgy = goals[i]
            sx, sy = starts[i]
            ax.plot(pgx, pgy, color=rnd[i], markersize=10, marker='X')
            ax.plot(sx, sy, color=rnd[i], markersize=10, marker='o')
            x, y = positions[i]
            ax.plot(x, y, color=rnd[i], markersize=5, marker='o')


        plt.show()


if __name__ == '__main__':
    sg = 1
    set = "sp3"
    success_failure_path = './mats_rst/sg{}/success_failure{}.mat'.format(sg,set)
    alg = Cadrl(alg_name="DuelingDDQN")
    alg.train(success_failure_path, sg, set, train=False, display_step=False, display_traj=False)
