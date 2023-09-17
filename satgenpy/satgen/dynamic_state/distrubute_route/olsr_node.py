import copy
import pickle

import ephem
import networkx as nx
import math
from .scheduler_node import Scheduler_node
from typing import Dict

from satgen.distance_tools import distance_m_between_satellites


class Olsr_node(Scheduler_node):
    def __init__(self,node_id,num_nodes,sat_net_graph_with_sat_and_gs, epoch,time_step, sim_duration):
        '''
        :param sat_name: 卫星名字
        :param line_1: 轨道两行数 第一行
        :param line_2: 轨道两行数 第二行
        '''
        super().__init__(epoch, time_step, sim_duration)


        self.update_sat_net_graph_with_sat_and_gs(sat_net_graph_with_sat_and_gs)
        self.point_to_all_node = []

        self.node_id = node_id
        self.num_nodes= num_nodes

        self.num_avail_neighbors =0
        self.avail_neighbors = {}
        '''
        :key node_id
        :var { "symmetry":yes/no; }
        
        '''
        self.mpr_set = []

        self.tc_table = {}
        '''
              :key node_id
              :var { "symmetry":Ture/False; }

        '''
        self.forwarding_table = {}
        self.forwarding_cost ={}
        for node_id in range(num_nodes):
            self.forwarding_table[node_id] = -1
            self.forwarding_cost[node_id] = math.inf

        self.seq = 0
        self.max_seq = 1000000000

        # hello_interval: 100 ms
        # fail_edges_update_interval = 2 s
        self.hello_interval = 0.1
        self.tc_broadcast_interval = 5
        self.slot_num_hello_interval = math.ceil(self.hello_interval/self.time_step)
        self.slot_num_tc_broadcast_interval = math.ceil(self.tc_broadcast_interval / self.time_step)
        self.slot_to_hello = 0
        self.slot_to_broadcast_tc = 0
        self.num_msg_rev = 0


    def get_neighbors(self):
        sym_neighbor = set()
        for neibghbor_id,neibghbor_msg in self.avail_neighbors.items():
            sym_neighbor.add(neibghbor_id)
        return sym_neighbor

    def hello(self):
        msg = {
            "type": "hello",
            "src_node_id": self.node_id,
            "msg_seq": self.get_seq(),
            "neighbor":self.get_neighbors()
        }
        for neighbor_id in self.sat_net_graph_with_sat_and_gs.neighbors(self.node_id):
            self.sent_msg(msg,neighbor_id)

    def deal_hello_msg(self, msg):
        src_node_id = msg["src_node_id"]
        src_neighbor = msg["neighbor"]
        if src_node_id in self.avail_neighbors:
            need_update_mpr = False
            if not self.avail_neighbors[src_node_id]["symmetry"] and self.node_id in src_neighbor:
                self.avail_neighbors[src_node_id]["symmetry"] = True
                need_update_mpr=True
            if self.avail_neighbors[src_node_id]["neighbor"] != src_neighbor:
                need_update_mpr = True
            self.avail_neighbors[src_node_id]["neighbor"] = copy.deepcopy(src_neighbor)
            self.avail_neighbors[src_node_id]["Ltime"] = self.curr_slot + 3 * self.slot_num_hello_interval
            if need_update_mpr:
                self.update_mpr()
        else:
            self.avail_neighbors[src_node_id] = {}
            if self.node_id in src_neighbor:
                self.avail_neighbors[src_node_id]["symmetry"] = True
            else:
                self.avail_neighbors[src_node_id]["symmetry"] = False
            self.avail_neighbors[src_node_id]["neighbor"] = copy.deepcopy(src_neighbor)
            self.avail_neighbors[src_node_id]["mpr"] = False
            self.avail_neighbors[src_node_id]["Ltime"] = self.curr_slot + 3 * self.slot_num_hello_interval
            if self.avail_neighbors[src_node_id]["symmetry"]:
                self.update_mpr()

    def hello_helper(self):
        if self.curr_slot == self.slot_to_hello:
            # 每次hello 之前，先进行 mpr 选举
            self.update_mpr()
            self.hello()
            self.slot_to_hello = self.slot_to_hello + self.slot_num_hello_interval

    def update_mpr(self):
        # print("mprmprmpmrapfjpao")
        mpr_set = set()
        one_hop_neighbors = self.get_neighbors()
        if one_hop_neighbors:
            two_hop_neighbors = set()
            for one_hop_neighbor in one_hop_neighbors:
                two_hop_neighbors.update(self.avail_neighbors[one_hop_neighbor]["neighbor"])
            # 去除一条邻居和本节点，剩余的才是二跳邻居
            two_hop_neighbors.difference_update(one_hop_neighbors)
            two_hop_neighbors.discard(self.node_id)
            while two_hop_neighbors:
                mpr_candidate = max(one_hop_neighbors, key=lambda x: len(two_hop_neighbors.intersection(self.avail_neighbors[x]["neighbor"])))
                mpr_set.add(mpr_candidate)
                # 删除已经被MPR覆盖的二跳邻居
                two_hop_neighbors.difference_update(self.avail_neighbors[mpr_candidate]["neighbor"])
            if mpr_set != self.mpr_set:
                self.mpr_set = mpr_set
                self.tc_broadcast()
                # self.need_tc_broadcast = True
        else:
            self.mpr_set = set()
            self.tc_broadcast()
            # self.need_tc_broadcast = True




    def tc_broadcast(self):
        msg = {
            "type": "tc",
            "src_node_id": self.node_id,
            "msg_seq": self.get_seq(),
            "mpr":copy.deepcopy(self.mpr_set),
            "pre_node_id": self.node_id,
        }
        for neighbor_id in self.sat_net_graph_with_sat_and_gs.neighbors(self.node_id):
            self.sent_msg(msg,neighbor_id)

    def deal_tc_msg(self,msg):
        src_node_id = msg["src_node_id"]
        msg_seq = msg["msg_seq"]
        mpr = copy.deepcopy(msg["mpr"])
        pre_node_id = msg["pre_node_id"]
        if (pre_node_id not in self.avail_neighbors) or (not self.avail_neighbors[pre_node_id]["symmetry"]):
            pass
        else:
            if (src_node_id not in self.tc_table) or (self.is_latest_seq(msg_seq,self.tc_table[src_node_id]["msg_seq"])):
                self.update_item_in_tc_table(src_node_id,msg_seq,mpr)
                # 如果本节点是上一节点的 mpr，则需要转发消息
                if ( pre_node_id in self.tc_table) and (self.node_id in self.tc_table[pre_node_id]["mpr"]):
                    msg = {
                        "type": "tc",
                        "src_node_id": src_node_id,
                        "msg_seq": msg_seq,
                        "mpr": copy.deepcopy(mpr),
                        "pre_node_id": self.node_id,
                    }
                    for neighbor_id in self.sat_net_graph_with_sat_and_gs.neighbors(self.node_id):
                        self.sent_msg(msg,neighbor_id)


    def update_item_in_tc_table(self,src_node_id,msg_seq,mpr):
        item  = {
            "src_node_id":src_node_id,
            "msg_seq":msg_seq,
            "mpr":mpr,
            "Ltime":self.curr_slot + 3 * self.slot_num_tc_broadcast_interval
        }
        if (src_node_id not in self.tc_table) or mpr!=self.tc_table[src_node_id]["mpr"]:
            self.need_update_forwarding_table = True
        self.tc_table[src_node_id] = item

    def update_forwarding_table(self):
        # print(self.curr_slot,self.node_id,"ss")
        topo = copy.deepcopy(self.sat_net_graph_with_sat_and_gs)
        for u,v,edge in topo.edges(data=True):
            edge['weight'] = math.inf
        for neighbor_id in self.avail_neighbors.keys():
            topo.add_edge(self.node_id, neighbor_id, weight=1)
        for node_id,tc_item in self.tc_table.items():
            for mpr_node_id in tc_item["mpr"]:
                topo.add_edge(node_id, mpr_node_id, weight=1)

        d_path = nx.single_source_dijkstra_path(topo,self.node_id)
        d_path_len = nx.single_source_dijkstra_path_length(topo,self.node_id)
        for i in range(self.num_nodes):
            if i != self.node_id:
                if d_path_len[i]!=math.inf:

                    self.forwarding_table[i] = d_path[i][1]
                    self.forwarding_cost[i] = d_path_len[i]
                else:
                    self.forwarding_table[i]=-1
                    self.forwarding_cost[i] = math.inf



    def update_sat_net_graph_with_sat_and_gs(self, sat_net_graph_with_sat_and_gs):
        self.sat_net_graph_with_sat_and_gs = copy.deepcopy(sat_net_graph_with_sat_and_gs)

    def get_seq(self):
        pre_seq = self.seq
        self.seq = (self.seq + 1) % self.max_seq
        return pre_seq

    def is_latest_seq(self, input_seq,orgin_seq):
        is_or_not = False
        if input_seq > orgin_seq and (input_seq - orgin_seq) <= (self.max_seq/2):
            is_or_not = True
        if input_seq < orgin_seq and (input_seq - orgin_seq) > (self.max_seq/2):
            is_or_not = True
        return is_or_not


    def sent_msg(self, msg, dst_node_id):
        if self.sat_net_graph_with_sat_and_gs[self.node_id][dst_node_id]["weight"] != math.inf:
            slot_cost = self.get_trans_slot_cost(self.sat_net_graph_with_sat_and_gs[self.node_id][dst_node_id]["weight"])
            if self.curr_slot + slot_cost < len(self.time_event_scheduler):
                self.point_to_all_node[dst_node_id].time_event_scheduler[self.curr_slot + slot_cost].append(msg)

    def deal_msg(self,msg):
        msg_type = msg["type"]
        if msg_type == "hello":
            self.deal_hello_msg(msg)
        elif msg_type == "tc":
            self.deal_tc_msg(msg)


    def broadcast_tc_helper(self):
        if self.curr_slot == self.slot_to_broadcast_tc:
            self.tc_broadcast()
            self.slot_to_broadcast_tc = self.slot_to_broadcast_tc + self.slot_num_tc_broadcast_interval

    def check_lifetime(self):
        fail_neighbors = []
        for neighbor_id,link_msg in self.avail_neighbors.items():
            if link_msg["Ltime"]<self.curr_slot:
                fail_neighbors.append(neighbor_id)
        for neighbor_id in fail_neighbors:
           del self.avail_neighbors[neighbor_id]
        fail_tc_item = []
        for node_id,tc_item in self.tc_table.items():
            if tc_item["Ltime"]<self.curr_slot:
                fail_tc_item.append(node_id)
        for node_id in fail_tc_item:
            del self.tc_table[node_id]

        if fail_neighbors:
            self.need_update_mpr = True
            self.need_update_forwarding_table = True

        if fail_tc_item:
            self.need_update_forwarding_table = True




    def process(self):
        self.need_update_mpr = False
        self.need_tc_broadcast = False
        self.need_update_forwarding_table = False
        for msg in self.time_event_scheduler[self.curr_slot]:
            self.deal_msg(msg)

        self.num_msg_rev = self.num_msg_rev + len(self.time_event_scheduler[self.curr_slot])
        self.time_event_scheduler[self.curr_slot] = []

        if self.need_update_mpr:
            self.update_mpr()

        if self.need_tc_broadcast:
            self.tc_broadcast()

        self.hello_helper()
        self.broadcast_tc_helper()
        if self.need_update_forwarding_table:
            self.update_forwarding_table()

        # 执行完当前时隙的任务后，更新本节点时隙
        self.update_curr_slot()

    def update_point_to_all_node(self,point_to_all_node):
        self.point_to_all_node = point_to_all_node









if __name__ == "__main__":

    epoch = ephem.Date("2000/1/1 00:00:00")
    time_step = 100
    sim_duration = 10
    satellites = []
    sat_net_graph_only_satellites_with_isls = nx.Graph()
    sat_net_graph_with_sat_and_gs =nx.Graph()

    from astropy.time import Time
    from astropy import units as u
    filename_tles = "tles.txt"
    with open(filename_tles, 'r') as f:
        num_orbs, num_sats_per_orbs = [int(n) for n in f.readline().split()]
        universal_epoch = None
        i = 0
        for sat_name in f:
            line_1 = f.readline()
            line_2 = f.readline()

            # Retrieve name and identifier
            name = sat_name
            sid = int(name.split()[1])
            if sid != i:
                raise ValueError("Satellite identifier is not increasing by one each line")
            i += 1

            epoch_year = line_1[18:20]
            epoch_day = float(line_1[20:32])
            epoch = Time("20" + epoch_year + "-01-01 00:00:00", scale="tdb") + (epoch_day - 1) * u.day
            if universal_epoch is None:
                universal_epoch = epoch
            if epoch != universal_epoch:
                raise ValueError("The epoch of all TLES must be the same")
            satellite = ephem.readtle(sat_name, line_1, line_2)


            # Finally, store the satellite information
            satellites.append(satellite)

    # 建立简单的卫星 Gird ，边的权值均为 1，待替换为实际距离
    shift_between_last_and_first = 8
    all_edges = []
    for i in range(num_orbs):
        for j in range(num_sats_per_orbs):
            sid = i * num_sats_per_orbs + j
            nid_next = i * num_sats_per_orbs + (j + 1) % num_sats_per_orbs
            if i == num_orbs - 1:
                nid_right = ((i + 1) % num_orbs) * num_sats_per_orbs + (
                            j + shift_between_last_and_first) % num_sats_per_orbs
            else:
                nid_right = ((i + 1) % num_orbs) * num_sats_per_orbs + j
            all_edges.append((sid,nid_next))
            all_edges.append((sid,nid_right))
            sat_net_graph_only_satellites_with_isls.add_edge(sid, nid_next, weight=1)
            sat_net_graph_only_satellites_with_isls.add_edge(sid, nid_right, weight=1)


    import random
    # 百分之坏边
    percentage = 0.01
    num_fail_edges = int(percentage * num_orbs * num_sats_per_orbs * 2)
    fail_edges = set(random.sample(all_edges, num_fail_edges))
    # for fail_edge in fail_edges:
    #     sat_net_graph_only_satellites_with_isls[fail_edge[0]][fail_edge[1]]["weight"] = math.inf

    #
    #
    # with open("gsls.pkl","rb") as f:
    #     gsls = pickle.load(f)
    #
    #
    #
    # num_nodes = len(satellites)+len(gsls)
    # sat_net_graph_with_sat_and_gs = copy.deepcopy(sat_net_graph_only_satellites_with_isls)
    # for node_id in range(len(gsls)):
    #     sat_net_graph_with_sat_and_gs.add_edge(node_id+len(satellites),0,weight=math.inf)
    #
    # for node_id in range(num_nodes):
    #     olsr_nodes = olsr_node(node_id, num_nodes, sat_net_graph_with_sat_and_gs, epoch, time_step, sim_duration)
    #
    olsr_nodes = []
    for node_id in range(len(satellites)):
        olsr_node = Olsr_node(node_id, len(satellites), sat_net_graph_only_satellites_with_isls, epoch, time_step, sim_duration)
        olsr_nodes.append(olsr_node)

    for olsr_node in olsr_nodes:
        olsr_node.update_point_to_all_node(olsr_nodes)


    num_time_slot = olsr_nodes[0].num_time_slot
    d_path = dict(nx.all_pairs_dijkstra_path(sat_net_graph_only_satellites_with_isls))
    d_path_len = dict(nx.all_pairs_dijkstra_path_length(sat_net_graph_only_satellites_with_isls))

    error = []
    error_2 = []
    import time


    for time_slot in range(num_time_slot):
        start = time.time()
        if time_slot == num_time_slot/2:
            for fail_edge in fail_edges:
                sat_net_graph_only_satellites_with_isls[fail_edge[0]][fail_edge[1]]["weight"] = math.inf

        print(time_slot,"kk")

        date = olsr_nodes[0].epoch + olsr_nodes[0].curr_slot *olsr_nodes[0].time_step * ephem.second
        for satellite in satellites:
            satellite.compute(date.datetime)
        for u,v,attrs in sat_net_graph_only_satellites_with_isls.edges(data=True):
            edge = (u,v)
            edge_weight = attrs["weight"]
            if edge_weight != math.inf:
                distance = distance_m_between_satellites(satellites[edge[0]], satellites[edge[1]], "2000/01/01 00:00:00", "2000/01/01 00:00:01")
                sat_net_graph_only_satellites_with_isls[edge[0]][edge[1]]["weight"] = distance

        for olsr_node in olsr_nodes:
            olsr_node.update_sat_net_graph_with_sat_and_gs(sat_net_graph_only_satellites_with_isls)
            olsr_node.process()
        stop = time.time()
        print(stop - start)

    print(olsr_nodes[0].forwarding_table)

