import copy
import ephem
from  scheduler_node import Scheduler_node
import numpy as np
import networkx as nx
import math
from satgen.tles import read_tles

class Sat_selector(Scheduler_node):
    def __init__(self,visible_times,satellites,num_orbs,num_sats_per_orbs,shift_between_last_and_first,plus_grid_graph,epoch,time_step, sim_duration):
        super().__init__(epoch,time_step, sim_duration)
        self.visible_times = visible_times
        self.satellites = satellites
        self.num_orbs = num_orbs
        self.num_sats_per_orbs = num_sats_per_orbs
        self.shift_between_last_and_first = shift_between_last_and_first
        self.plus_grid_graph = plus_grid_graph
        self.num_sats = self.num_orbs * self.num_sats_per_orbs
        self.num_gs = len(visible_times)
        self.gsls = {}
        for gid in range(self.num_gs):
            self.gsls[gid] = -1

        self.comm_demand =  np.ones((self.num_gs, self.num_gs), dtype=int)

    def determine_satellite_direction(self,satellite, current_time):
        # 计算卫星的当前位置
        satellite.compute(current_time)

        # 获取当前纬度和1秒后的纬度
        current_latitude = satellite.sublat
        next_time = ephem.Date(current_time + ephem.second)
        satellite.compute(next_time)
        next_latitude = satellite.sublat

        # 确定卫星飞行方向
        if next_latitude >= current_latitude:
            return "north"
        elif next_latitude < current_latitude:
            return "south"

    def classificate(self, sats_visible,current_time):
        northbound_satellites = []
        southbound_satellites = []

        for sid in sats_visible:
            satellite = self.satellites[sid]
            direction = self.determine_satellite_direction(satellite, current_time)
            if direction == "north":
                northbound_satellites.append(sid)
            elif direction == "south":
                southbound_satellites.append(sid)

        return northbound_satellites, southbound_satellites

    def get_visible_time_left(self,sid,gid,current_time):
        visible_time = self.visible_times[gid][sid]
        if current_time >= visible_time[0] and current_time <= visible_time[1]:
            visible_time_left = (visible_time[1] - current_time) / ephem.second
            return  visible_time_left
        else:
            return 0

    def get_sats_visible(self,gid,min_RST,current_time):
        sats_visible = []
        for sid,visible_time in self.visible_times[gid].items():
            if current_time >= visible_time[0] and current_time <= visible_time[1]:
                visible_time_left = (visible_time[1] - current_time)/ephem.second
                if visible_time_left>=min_RST*ephem.second:
                    sats_visible.append(sid)
        return sats_visible

    def init_gsl_current_time(self,min_RST,current_time):
        sats_visible_of_all_gs = []
        sid_north_visible_of_all_gs = []
        index_north_visible_of_all_gs = []
        sid_south_visible_of_all_gs = []
        index_south_visible_of_all_gs = []

        self.center_index_north_visible_of_all_gs= []
        self.center_index_south_visible_of_all_gs = []

        for src_gid in range(self.num_gs):
            sats_visible = self.get_sats_visible(src_gid,min_RST,current_time)
            sats_visible_of_all_gs.append(sats_visible)
            northbound_satellites, southbound_satellites = self.classificate(sats_visible,current_time)

            sid_north_visible_of_all_gs.append(northbound_satellites)
            index_north_visible_of_all_gs.append([self.get_index_from_sid(sid) for sid in northbound_satellites])

            sid_south_visible_of_all_gs.append(southbound_satellites)
            index_south_visible_of_all_gs.append([self.get_index_from_sid(sid) for sid in southbound_satellites])

            if index_north_visible_of_all_gs[src_gid]:
                x_center_north = sum( index[0] for index in index_north_visible_of_all_gs[src_gid] ) / len(index_north_visible_of_all_gs[src_gid])
                y_center_north = sum(index[1] for index in index_north_visible_of_all_gs[src_gid])  / len(index_north_visible_of_all_gs[src_gid])
                self.center_index_north_visible_of_all_gs.append([x_center_north,y_center_north])
            else:
                self.center_index_north_visible_of_all_gs.append([])

            if index_south_visible_of_all_gs[src_gid]:
                x_center_south = sum(index[0] for index in index_south_visible_of_all_gs[src_gid]) / len(index_south_visible_of_all_gs[src_gid])
                y_center_south = sum(index[1] for index in index_south_visible_of_all_gs[src_gid]) / len(index_south_visible_of_all_gs[src_gid])
                self.center_index_south_visible_of_all_gs.append([x_center_south,y_center_south])
            else:
                self.center_index_south_visible_of_all_gs.append([])

        self.selects = [-1] * self.num_gs
        # print(self.selects)
        self.gsls_index = [-1] * self.num_gs
        self.select_cost = math.inf
        d_path_len = dict(nx.all_pairs_dijkstra_path_length(self.plus_grid_graph))

        # 用于遍历所有可能的南北组合，获得最低的全局跳数对应组合
        def generate_current_selects(a, b, i, current_selects):
            if i == len(a)-1:
                cost = 0
                gsls_index = [-1] * self.num_gs
                # print(sum(current_selects),current_selects)
                for gid,direction in enumerate(current_selects):
                    if direction == 0 :
                        if self.center_index_north_visible_of_all_gs[gid]:
                            gsls_index[gid] = self.center_index_north_visible_of_all_gs[gid]
                        else:
                            gsls_index[gid] = -1
                    else:
                        if self.center_index_south_visible_of_all_gs[gid]:
                            gsls_index[gid] = self.center_index_south_visible_of_all_gs[gid]
                        else:
                            gsls_index[gid] = -1

                for src_index in gsls_index:
                    for dst_index in gsls_index:
                        if src_index!=-1 and dst_index!= -1:
                            # print(src_index,dst_index,"lls")
                            # print(int(self.get_sid_from_index(src_index[0], src_index[1])),int(self.get_sid_from_index(dst_index[0],dst_index[1])),"sss")

                            cost = cost + d_path_len[int(self.get_sid_from_index(src_index[0],src_index[1]))][int(self.get_sid_from_index(dst_index[0],dst_index[1]))]
                if cost<self.select_cost:
                    self.select_cost = cost
                    self.selects = current_selects
                    self.gsls_index = gsls_index
                    # print(f"cost update:{self.select_cost},selects update:{self.selects}")

                return

            # 从a[i]中选择
            # print(len(current_selects),len(a))
            # print(i)
            current_selects[i] = a[i]
            generate_current_selects(a, b, i + 1, current_selects)

            # 从b[i]中选择
            current_selects[i] = b[i]
            generate_current_selects(a, b, i + 1, current_selects)

        # 辅助列表a和b
        a = [0] * self.num_gs
        b = [1] * self.num_gs
        # 初始化当前组合和
        current_sum = 0

        # 初始化当前组合列表
        current_selects = [0] * len(a)
        # 获得最低的全局跳数对应组合
        generate_current_selects(a, b, 0, current_selects)

        for src_gid in range(self.num_gs):
            temp_cost = math.inf
            temp_gsl = -1
            if self.selects[src_gid] == 0:
                sats_select = sid_north_visible_of_all_gs[src_gid]
            else:
                sats_select = sid_south_visible_of_all_gs[src_gid]
            if sats_select:
                for sid in sats_select:
                    cost = 0
                    for dst_gid in  range(self.num_gs):
                        index = self.gsls_index[dst_gid]
                        if index!=-1:
                            index = (int(index[0]), int(index[1]))
                            cost =cost+d_path_len[sid][self.get_sid_from_index(index[0],index[1])]
                    if cost < temp_cost:
                        temp_cost = cost
                        temp_gsl = sid

            self.gsls[src_gid] = temp_gsl
            if temp_gsl!=-1:
                self.gsls_index[src_gid] = self.get_index_from_sid(temp_gsl)
            else:
                self.gsls_index[src_gid] = -1

    def update_gsl(self,min_RST,type = "change"):
        current_time = ephem.Date(self.epoch + self.curr_slot * self.time_step * ephem.second)
        count_shift = {}
        shift_index_to_pre_sat = [[]]*self.num_gs
        if type == "change":
            for src_gid in range(self.num_gs):
                sats_visible = self.get_sats_visible(src_gid, min_RST, current_time)
                northbound_satellites, southbound_satellites = self.classificate(sats_visible, current_time)
                if self.selects[src_gid] == 0:
                    sids_to_selects = northbound_satellites
                else:
                    sids_to_selects = southbound_satellites
                for sid in sids_to_selects:
                    x,y = self.get_index_from_sid(sid)
                    x_shift = (x - self.gsls_index[src_gid][0]+self.num_orbs)%self.num_orbs
                    y_shift = (y - self.gsls_index[src_gid][1] + self.num_sats_per_orbs) % self.num_sats_per_orbs
                    if (x_shift,y_shift) in count_shift:
                        count_shift[(x_shift, y_shift)] = count_shift[(x_shift, y_shift)] + 1
                    else:
                        count_shift[(x_shift, y_shift)] = 1
                    shift_index_to_pre_sat[src_gid].append((x_shift,y_shift))
            count = 0
            if not count_shift:
                print("ddddd")
            else:
                print("lllll",count_shift)
            for (x_shift, y_shift) in count_shift.keys():
                if count_shift[(x_shift, y_shift)] > count:
                    count = count_shift[(x_shift, y_shift)]
                    ( most_x_shift, most_y_shift) = (x_shift, y_shift)


            for src_gid in range(self.num_gs):
                if ( most_x_shift, most_y_shift) in shift_index_to_pre_sat[src_gid]:
                    x = (x_shift + self.gsls_index[src_gid][0] + self.num_orbs) % self.num_orbs
                    y = (y_shift + self.gsls_index[src_gid][1] + self.num_sats_per_orbs) % self.num_sats_per_orbs
                    new_gsl = self.get_sid_from_index(x,y)
                    self.gsls[src_gid] = new_gsl
                    self.gsls_index[src_gid] = (x,y)
                elif shift_index_to_pre_sat[src_gid]:
                    temp = math.inf
                    for (x_shift, y_shift) in shift_index_to_pre_sat[src_gid]:
                        if abs(x_shift-most_x_shift) + abs(y_shift-most_y_shift) < temp:
                            temp = abs(x_shift-most_x_shift) + abs(y_shift-most_y_shift)
                            x = (x_shift + self.gsls_index[0] + self.num_orbs) % self.num_orbs
                            y = (y_shift + self.gsls_index[1] + self.num_sats_per_orbs) % self.num_sats_per_orbs
                    self.gsls[src_gid] = new_gsl
                    self.gsls_index[src_gid] = (x, y)
                else:
                    self.gsls[src_gid] = -1
                    self.gsls_index[src_gid] = -1

        elif type =="init":
            self.init_gsl_current_time(min_RST,current_time)

    def get_index_from_sid(self,sid):
        orb_id = sid // self.num_sats_per_orbs
        sat_id_in_orb = sid % self.num_sats_per_orbs
        return orb_id,sat_id_in_orb

    def get_sid_from_index(self,orb_id,sat_id_in_orb):
        return orb_id*self.num_sats_per_orbs+sat_id_in_orb

    def get_neighbor_index(self,sid,x,y):
        orb_id, sat_id_in_orb = self.get_index_from_sid(sid)
        neighbor_orb_id = orb_id+x
        if neighbor_orb_id >= self.num_orbs:
            neighbor_orb_id = neighbor_orb_id - self.num_orbs
            neighbor_sat_id_in_orb = (sat_id_in_orb + y + self.shift_between_last_and_first)%self.num_sats_per_orbs
        elif neighbor_orb_id < 0:
            neighbor_orb_id = neighbor_orb_id + self.num_orbs
            neighbor_sat_id_in_orb = (sat_id_in_orb + y - self.shift_between_last_and_first) % self.num_sats_per_orbs
        else:
            neighbor_sat_id_in_orb = (sat_id_in_orb + y) % self.num_sats_per_orbs
        return neighbor_orb_id,neighbor_sat_id_in_orb

    def add_gsl_update_event(self,gid,gsl_to_update,slot_to_update_gsl):
        if slot_to_update_gsl < self.num_time_slot:
            msg ={
                "type": "gsl_update",
                "gid": gid,
                "gsl_to_update": gsl_to_update
            }
            self.time_event_scheduler[slot_to_update_gsl].append(msg)

    def deal_msg(self,msg):
        if msg["type"] == "gsl_update":
            gid = msg["gid"]
            gsl_to_update = msg["gsl_to_update"]
            sid,visible_slot_left = self.update_sats_a_gs_linked_to(gid,gsl_to_update)
            if sid!= -1:
                self.add_gsl_update_event(gid,gsl_to_update,self.curr_slot+visible_slot_left)
            else:
                self.add_gsl_update_event(gid,gsl_to_update,self.curr_slot+1)

    def process(self):
        need_init = False
        current_time = ephem.Date(self.epoch + self.curr_slot * self.time_step * ephem.second)
        for gsl in self.gsls.values():
            if gsl == -1:
                need_init =True
        if need_init:
            print(f"init {self.curr_slot}")
            self.update_gsl(0.150, type="init")
        else:
            for gid,gsl in enumerate(self.gsls):
                if self.get_visible_time_left(gsl,gid,current_time) <= 0:
                    print(f"change {self.curr_slot},{gid},{gsl}")
                    self.update_gsl(0.150, type="change")
                    break
        self.curr_slot = self.curr_slot + 1


if __name__ == "__main__":
    pass
    import pickle
    with open('visible_times_15.pkl', 'rb') as f:
        visible_times = pickle.load(f)
    num_orbs = 72
    num_sats_per_orbs = 18
    shift_between_last_and_first = 8
    epoch = ephem.Date("2000/0/0 00:00:00")
    time_step = 100
    sim_duration = 1000

    tles = read_tles("tles.txt")
    satellites = tles["satellites"]

    plus_grid_graph = nx.Graph()
    for i in range(num_orbs):
        for j in range(num_sats_per_orbs):
            sid = i * num_sats_per_orbs + j
            nid_next = i * num_sats_per_orbs + (j + 1) % num_sats_per_orbs
            if i == num_orbs - 1:
                nid_right = ((i + 1) % num_orbs) * num_sats_per_orbs + (
                            j + shift_between_last_and_first) % num_sats_per_orbs
            else:
                nid_right = ((i + 1) % num_orbs) * num_sats_per_orbs + j
            plus_grid_graph.add_edge(sid, nid_next, weight=1)
            plus_grid_graph.add_edge(sid, nid_right, weight=1)

    sat_selector = Sat_selector(visible_times,satellites,num_orbs,num_sats_per_orbs,shift_between_last_and_first,plus_grid_graph,epoch,time_step, sim_duration)
    # for gid,gsl in sat_selector.gsls.items():
    #     # 对每个 gs 的 gsl 更新两次
    #     for i in range(2):
    #         sat_selector.add_gsl_update_event(gid,i,0)
    gsls_of_all_case = []
    switch = []
    for i in range(sat_selector.num_time_slot):
        print(i)
        sat_selector.process()
        if (not gsls_of_all_case) or gsls_of_all_case[-1]!=sat_selector.gsls:
            gsls_of_all_case.append( copy.deepcopy(sat_selector.gsls) )
            switch.append(i)
        print(sat_selector.gsls)


