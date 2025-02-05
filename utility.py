
"""
This file includes some supportive functions.
"""


import numpy as np
import random
import math
import itertools



class Utility():
    """
    This class includes the supportive functions.
    """
    def rotate_state(self, env,state,  max_nodes=5, max_obv_region=10,rotate=False,change_node=False):

        st, snt, stt = state
        px, py, H, vx, vy, radius, pgx, pgy, v_pref_mag, theta, EE, charge = st
        # px, py, H, vx, vy, pgx, pgy, theta, EE = st
        if rotate:  # False
            pgx_r, pgy_r = pgx - px, pgy - py
            dg = math.sqrt(pgx_r ** 2 + pgy_r ** 2)
            rot = self.get_angle(pgx_r, pgy_r)  # angel is in between [0,2pi)
            vx_r = vx * math.cos(rot) + vy * math.sin(rot)
            vy_r = vy * math.cos(rot) - vx * math.sin(rot)
            theta_r = (theta - rot) % (2 * math.pi)
            # st_r = [vx_r, vy_r, dg, radius, v_pref_mag, theta_r , stt, EE]
            st_r = [vx_r, vy_r, dg, theta_r, stt, EE]


            xn, yn, snt_r = [], [], []
            for io in range(len(snt)):
                xn.append(snt[io][0])
                yn.append(snt[io][1])
            _, _, new_index_n = self.get_closest(px, py, xn, yn,len(snt))  # This part is like to sort the distance from other agents
            if change_node:
                for i in range(len(snt)):
                    bsx, bsy,pak = snt[i]
                    if pak >0:
                        mag = math.sqrt((bsx - px) ** 2 + (bsy - py) ** 2)
                        ang = self.get_angle(bsx - px, bsy - py)
                        # indicator = int(mag <= env.nodes_r)
                        sinr = env.get_sinr(px,py,mag)
                        indicator = int(sinr>=env.Tsinr)
                        # print('dis {}, sinr is {}, and indicator is {}'.format(mag,sinr,indicator))
                        snt_r.append([mag, ang, pak,sinr,indicator])
                if len(snt_r) >= max_nodes:
                    snt_r = snt_r[0:max_nodes]
                else:
                    ls = max_nodes - len(snt_r)
                    for _ in range(ls):
                        snt_r.append([0, 0, 0, 0, 0])
            else:
                for i in range(len(snt)):
                    bsx, bsy, pak = snt[i]
                    mag = math.sqrt((bsx - px) ** 2 + (bsy - py) ** 2)
                    ang = self.get_angle(bsx - px, bsy - py)
                    # indicator = int(mag <= env.nodes_r)
                    sinr = env.get_sinr(px,py,mag)
                    indicator = int(sinr >= env.Tsinr)
                    # print('dis {}, sinr is {}, and indicator is {}'.format(mag,sinr,indicator))
                    snt_r.append([mag, ang, pak, sinr, indicator])
            return [st_r, snt_r]
        else:
            # put the agent at the origin, but the xy coordinate is the global xy coordinate.
            pgx_r, pgy_r = -px,-py
            dg = math.sqrt(pgx_r ** 2 + pgy_r ** 2)
            ag = self.get_angle(pgx_r, pgy_r)  # angel is in between [0,2pi)
            is_uncharge,need_upload,need_charge = 0,0,0
            if charge <= 0:
                is_uncharge = 1
            if env.reward_aoi >=100:
                need_upload = 1
            if charge <= 20000:
                need_charge = 1
            # st_r = [px, py, vx, vy, radius, dg, ag, v_pref_mag, theta, stt]
            st_r = [vx, vy, -px, -py, dg, ag, theta, stt, EE, need_charge, is_uncharge,
                    env.reward_aoi / 100, need_upload]
            nodes_idx, sort_nodes_idx = [], []
            sort_snt = []
            xn, yn, snt_r = [], [], []
            for io in range(len(snt)):
                xn.append(snt[io][0])
                yn.append(snt[io][1])

            if change_node:         # True


                for i in range(len(snt)):
                    bsx, bsy, pak, timestamp = snt[i]
                    num_pak = sum(pak)
                    if num_pak >0:
                        avg_time = sum(timestamp)/len(timestamp)
                        bsx_r, bsy_r = bsx-px, bsy - py
                        mag = math.sqrt((bsx - px) ** 2 + (bsy - py) ** 2)
                        ang = self.get_angle(bsx - px, bsy - py)
                        # indicator = int(mag <= env.nodes_r)
                        sinr = env.get_sinr(px,py,mag,)
                        indicator = int(sinr >= env.Tsinr)
                        nodes_idx.append(i)
                        snt_r.append([bsx_r, bsy_r, mag, ang, num_pak, sinr, indicator, avg_time])
                nodes_info = list(zip(nodes_idx, snt_r))

                if len(snt_r) >= max_nodes:
                    sorted_nodes_info = sorted(nodes_info, key=lambda x: x[1][2])[:10]
                    # 输出排序后的节点信息
                    for node_id, node_info in sorted_nodes_info:
                        sort_nodes_idx.append(node_id)
                        sort_snt.append(node_info)
                else:
                    ls = max_nodes-len(snt_r)
                    sorted_nodes = sorted(nodes_info, key=lambda x: x[1][2])
                    for node_id, node_info in sorted_nodes:
                        sort_nodes_idx.append(node_id)
                        sort_snt.append(node_info)
                    for _ in range(ls):
                        sort_snt.append([0, 0, 0, 0, 0, 0, 0, 0])
            else:
                for i in range(len(snt)):
                    bsx, bsy, pak = snt[i]
                    bsx_r, bsy_r = bsx - px, bsy - py
                    mag = math.sqrt((bsx - px) ** 2 + (bsy - py) ** 2)
                    ang = self.get_angle(bsx - px, bsy - py)
                    # indicator = int(mag <= env.nodes_r)
                    sinr = env.get_sinr(px,py,mag)
                    indicator = int(sinr >= env.Tsinr)
                    snt_r.append([bsx_r, bsy_r, mag, ang, pak, sinr, indicator])
            return [st_r, sort_snt], sort_nodes_idx, sort_snt


    def get_closest(self,px,py,xarray,yarray,max_num):
        import heapq
        dis = list(np.sqrt((np.array(xarray) - px) ** 2 + (np.array(yarray) - py) ** 2))
        min_number = heapq.nsmallest(max_num, dis)
        xarray_new, yarray_new, min_index = [], [], []
        for t in min_number:
            index = dis.index(t)
            dis[index] = 0
            xarray_new.append(xarray[index])
            yarray_new.append(yarray[index])
            min_index.append(index)
        return xarray_new, yarray_new, min_index

    def get_state_mean_var(self,pairs):
        st,sn = [],[]
        for i in range(len(pairs)):
            s = np.squeeze(pairs[i][0])
            str,sor,snr = self.rotate_state(s)
            st.append(str)
            sn.append(snr)
        stm, sttd, snm, snd = list(np.mean(st, axis=0)), list(np.std(st, axis=0)), list(np.mean(sn, axis=0)), list(np.std(sn, axis=0))
        mean = [stm, snm]
        std = [sttd, snd]
        return mean, std

    def standardize_state(self, state, mean, std):
        state_s =[]
        for i in range(len(state)):
            st_s = self.standardize_state_sp(state[i], mean[i], std[i])
            state_s.append(st_s)
        return state_s

    def standardize_state_sp(self, state, mean, std):
        nstd = []
        if isinstance(std[0], list):
            for l in range(len(std)):
                nstd.append([])
                for it in std[l]:
                    if it <= 1e-6: it=1
                    nstd[l].append(it)
            nt = []

            for i in range(len(state)):
                if (np.array(state[i]) <= 0).all():
                    nt += state[i]
                else:
                    ns = (np.array(state[i]) - np.array(mean[i])) / np.array(nstd[i])
                    nt += list(ns)
        else:
            for it in std:
                if it <= 1e-6: it = 1
                nstd.append(it)

            ns = (np.array(state) - np.array(mean)) / np.array(nstd)
            nt = np.reshape(ns, (1, -1))
            nt = list(np.squeeze(nt))
        return nt

    def get_state_max_min(self,pairs):
        state = []
        for i in range(len(pairs)):
            s = np.squeeze(pairs[i][0])
            sr = self.rotate_state(s)
            state.append(sr)
        return list(np.max(state, axis=0)), list(np.min(state, axis=0))

    def normalize_state(self, state,mean,maxx,minn):
        rg = list((np.array(maxx)-np.array(minn)))
        nrg= []
        for it in rg:
            if it==0: it=1
            nrg.append(it)
        ns = (np.array(state)- np.array(mean) )/ np.array(nrg)
        return list((ns-0.5)/0.5)

    def z_score(self, x, axis):
        x = np.array(x).astype(float)
        # xr = np.rollaxis(x, axis=axis)
        xr = x
        xr -= np.mean(x, axis=axis)
        xr /= np.std(x, axis=axis)
        return xr


    def get_angle(self, x,y):
        """
        Give a vector (x,y), get its angle. All angles are in [0,2*pi)
        """
        return (math.atan2(y,x) + 2*math.pi) % (2*math.pi)

    def get_angle_diff(self, ag1, ag2):
        """
        Given two angles, get their difference. All angle_diff are in [0, pi)
        """
        abb = np.abs(ag1-ag2)
        if abb > math.pi:
            abb = 2*math.pi - abb
        return abb




# if __name__ == '__main__':
#     agent = Agent()
#     actions = agent.build_action_space(1)
#     print(actions)












