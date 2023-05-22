import math
import networkx as nx

num_orbs = 72
num_sats_per_orbs = 18
G = nx.Graph()

class Sat_wiht_cthulhu_Grid:

    # 字典键值为目标节点
    # 值为 [path,waypoint],waypoint=-1 时表示有直达路径
    # 值 [-1,-1]，表示不可达，无路径



    # G 网格中的边，必须严格遵循每个节点有四条边，若相邻节点间不可达，则必须为 math.inf
    def __init__(self, sid, num_orbs, num_sats_per_orbs, graph_cal_by_tles,shift_between_last_and_first):
        self.sid = sid
        self.num_orbs = num_orbs
        self.num_sats_per_orbs = num_sats_per_orbs
        self.orb_id = sid // num_sats_per_orbs
        self.sat_id_in_orb = sid % num_sats_per_orbs
        self.graph_cal_by_tles = graph_cal_by_tles
        self.shift_between_last_and_first = shift_between_last_and_first
        self.cthulhu_Grid_node_index = {}
        self.cthulhu_Grid = {}

        # 构建克苏鲁图的索引：以当前节点为中心，5*5 范围内的所有节点
        for i in range(-2,3):
            for j in range(-2,3):
                self.cthulhu_Grid_node_index[(i,j)] = self.get_sid_by_relative_hop(i, j)

        self.set_cthulhu_Grid()

    # 获得以当前节点为中心，相对位置为（i，j）的节点的编号
    def get_sid_by_relative_hop(self, hor, vel):
        sid_by_relative_hop_i = (self.orb_id+hor)%self.num_orbs
        if (self.orb_id+hor)>self.num_orbs-1:
            sid_by_relative_hop_j = (self.sat_id_in_orb + vel + self.shift_between_last_and_first ) % self.num_sats_per_orbs
        elif (self.orb_id+hor)<0:
            sid_by_relative_hop_j = (self.sat_id_in_orb + vel - self.shift_between_last_and_first) % self.num_sats_per_orbs
        else:
            sid_by_relative_hop_j = (self.sat_id_in_orb + vel) % self.num_sats_per_orbs

        return sid_by_relative_hop_i*self.num_sats_per_orbs + sid_by_relative_hop_j

    def set_cthulhu_Grid(self):
        # 以 cthulhu_Grid_node_index 的值建立子图
        self.subgraph = self.graph_cal_by_tles.subgraph( list(self.cthulhu_Grid_node_index.values()) )
        d_path_len = dict(nx.all_pairs_dijkstra_path_length(self.subgraph))
        d_path = dict(nx.all_pairs_dijkstra_path(self.subgraph))

        # 更新以当前节点为中心，3*3 网格内节点的路由
        for i in range(-1,2):
            for j in range(-1,2):
                if (abs(i)+abs(j)) !=0:
                    dst_sid = self.get_sid_by_relative_hop(i, j)
                    if d_path_len[self.sid][dst_sid] != math.inf:
                        self.cthulhu_Grid[dst_sid] = [d_path[self.sid][dst_sid],d_path_len[self.sid][dst_sid],dst_sid]

                    else:
                        # 节点不可达
                        self.cthulhu_Grid[dst_sid] = [-1,math.inf,-1]

        # 更新以当前节点为中心，5*5 网格边缘节点的路由
        candidate_node_list_five={
            # 候选节点若不处于同行和同列，优先考虑目标节点 3*3 网格内节点，否则候选下一跳必须为同列或同行的节点
            (0, 2): [(0, 1)],
            (1, 2): [(0, 1), (1, 0)],
            (2, 2): [(0, 1), (1, 0)],
            (2, 1): [(1, 0), (0, 1)],
            (2, 0): [(1, 0)],
            (2, -1): [(1, 0), (0, -1)],
            (2, -2): [(1, 0), (0, -1)],
            (1, -2): [(0, -1), (1, 0)],
            (0, -2): [(0, -1)],
            (-1, -2): [(0, -1), (-1, 0)],
            (-2, -2): [(0, -1), (-1, 0)],
            (-2, -1): [(-1, 0), (0, -1)],
            (-2, 0): [(-1, 0)],
            (-2, 1): [(-1, 0), (0, 1)],
            (-2, 2): [(-1, 0), (0, 1)],
            (-1, 2): [(0, 1), (-1, 0)],
        }
        for i in range(-2,3):
            for j in range(-2,3):
                # 任意相对坐标为 2，说明在 5*5 网格边缘
                if abs(i) == 2 or abs(j) == 2:
                    dst_sid = self.get_sid_by_relative_hop(i, j)
                    # 默认不可达
                    self.cthulhu_Grid[dst_sid] = [-1,math.inf, -1]
                    if d_path_len[self.sid][dst_sid] != math.inf:
                        self.cthulhu_Grid[dst_sid] = [d_path[self.sid][dst_sid],d_path_len[self.sid][dst_sid], dst_sid]
                    else:
                        for candidate_node_i,candidate_node_j in candidate_node_list_five[(i,j)]:
                            candidate_node_id = self.get_sid_by_relative_hop(candidate_node_i, candidate_node_j)
                            if d_path_len[self.sid][candidate_node_id] != math.inf:
                                self.cthulhu_Grid[dst_sid] = [d_path[self.sid][candidate_node_id], d_path_len[self.sid][candidate_node_id],candidate_node_id]
                                break
        # 建立全网节点
        candidate_node_list_far = {
            # 候选节点若不处于同行和同列，优先考虑目标节点 3*3 网格内节点，否则候选下一跳必须为同列或同行的节点
            (0, 2): [(0, 1), (1, 1), (-1, 1), (2, 2), (-2, 2)],
            (1, 2): [(0, 1), (1, 0), (-1, 2), (2, 1)],
            (2, 2): [(0, 1), (1, 0), (-1, 2), (2, -1)],
            (2, 1): [(1, 0), (0, 1), (2, -1), (1, 2)],
            (2, 0): [(1, 0), (1, 1), (1, -1), (2, 2), (2, -2)],
            (2, -1): [(1, 0), (0, -1), (2, 1), (1, -2)],
            (2, -2): [(0, -1), (1, 0), (-1, -2), (2, 1)],
            (1, -2): [(0, -1), (1, 0), (-1, -2), (2, -1)],
            (0, -2): [(0, -1), (1, -1), (-1, -1), (2, -2), (-2, -2)],
            (-1, -2): [(0, -1), (-1, 0), (1, -2), (-2, -1)],
            (-2, -2): [(0, -1), (-1, 0), (1, -2), (-2, 1)],
            (-2, -1): [(-1, 0), (0, -1), (-2, 1), (-1, -2)],
            (-2, 0): [(-1, 0), (-1, 1), (-1, -1), (-2, 2), (-2, -2)],
            (-2, 1): [(-1, 0), (0, 1), (-2, -1), (-1, 2)],
            (-2, 2): [(0, 1), (-1, 0), (1, 2), (-2, -1)],
            (-1, 2): [(0, 1), (-1, 0), (1, 2), (-2, 1)],
        }

        index_i = self.num_orbs//2
        index_j = self.num_sats_per_orbs//2
        for i in range(-index_i,index_i+self.num_orbs%2):
            for j in  range(-index_j,index_j+self.num_sats_per_orbs%2):
                # 只计算不在 5*5 网格内的
                if abs(i)>2 or abs(j)>2:
                    dst_sid = self.get_sid_by_relative_hop(i, j)
                    # 默认不可达
                    self.cthulhu_Grid[dst_sid] = [-1, math.inf, -1]
                    _i = i if abs(i)<=2 else int(math.copysign(2,i))
                    _j = j if abs(j) <= 2 else int(math.copysign(2, j))
                    for candidate_node_i,candidate_node_j in candidate_node_list_far[(_i,_j)]:
                        candidate_node_id = self.get_sid_by_relative_hop(candidate_node_i, candidate_node_j)
                        if d_path_len[self.sid][candidate_node_id] != math.inf:
                            self.cthulhu_Grid[dst_sid] = [d_path[self.sid][candidate_node_id], d_path_len[self.sid][candidate_node_id], candidate_node_id]
                            break

    # 获得到某点的子路径
    # 若直达则返回路径，不可达返回[-1,math.inf,-1]，其他情况返回子路径和航路点
    def get_sub_path_to_dst_sat(self,dst_sid:int):
        sub_path,sub_path_len,waypoint = self.cthulhu_Grid[dst_sid]
        if sub_path!=-1:
            return sub_path,sub_path_len,waypoint
        else:
            return [-1,math.inf,-1]

    def update_cthulhu_Grid(self,dst_sid,sub_path,sub_path_len):
        self.cthulhu_Grid[dst_sid] = [sub_path,sub_path_len,-1]



class Sat_network_routing_wiht_cthulhu_Grid:

    # 初始化，建立 cthulhu_Grid
    def __init__(self, num_orbs, num_sats_per_orbs, graph_cal_by_tles:nx.Graph,shift_between_last_and_first):
        self.num_update = 0
        self.num_orbs = num_orbs
        self.num_sats_per_orbs = num_sats_per_orbs
        self.graph_cal_by_tles = graph_cal_by_tles
        self.shift_between_last_and_first = shift_between_last_and_first
        self.cthulhu_Grid = {}
        for orb_id in range(self.num_orbs):
            for sat_id_in_orb in range(self.num_sats_per_orbs):
                sid = orb_id*self.num_sats_per_orbs + sat_id_in_orb
                self.cthulhu_Grid[sid] = Sat_wiht_cthulhu_Grid(sid, self.num_orbs, self.num_sats_per_orbs,self.graph_cal_by_tles,self.shift_between_last_and_first)

    # 模拟一次点对点寻路由
    # 寻路由同时，会更新航路点，以避免重复路径
    def path_a_to_b(self, a:int, b: int):
        current = a
        waypoint = a
        # path 默认不包含当前节点
        path = []
        path_len = 0
        while current!=b:
            sub_path,sub_path_len,waypoint = self.cthulhu_Grid[current].get_sub_path_to_dst_sat(b)
            if sub_path!=-1:
                path.extend(sub_path[:-1])
                path_len =path_len + sub_path_len
                current = sub_path[-1] # 或者 current = waypoint
            else:
                path_len = math.inf
                path.append(current)
                break
        if current == b:
            path.append(current)
        return  path,path_len