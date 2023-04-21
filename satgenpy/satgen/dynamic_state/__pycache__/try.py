import networkx as nx

# 创建一个有向图
G = nx.DiGraph()

# 添加节点和边
G.add_edge('A', 'B', weight=4)
G.add_edge('B', 'C', weight=8)
G.add_edge('C', 'D', weight=7)
G.add_edge('D', 'E', weight=9)
G.add_edge('E', 'F', weight=10)
G.add_edge('F', 'G', weight=2)
G.add_edge('G', 'H', weight=1)
G.add_edge('H', 'I', weight=7)
G.add_edge('I', 'J', weight=2)
G.add_edge('J', 'K', weight=4)
G.add_edge('K', 'L', weight=9)
G.add_edge('L', 'M', weight=14)
G.add_edge('M', 'N', weight=10)
G.add_edge('N', 'O', weight=2)
G.add_edge('O', 'P', weight=1)
G.add_edge('P', 'A', weight=8)
G.add_node('Z')

# 定义起点和终点
start = 'A'
end = 'P'

# 使用 A* 算法寻找最短路径
path = nx.astar_path(G, start, end, heuristic=None, weight='weight')

print(path)
# 打印路径
print(" -> ".join(path))

# print(path[1])

# G1 = nx.Graph()
# G1.add_node(3)
# G1.add_edge(0,1,weight=3)
# G1.add_edge(1,2,weight=3)
# path1 = nx.astar_path(G1,0,0,heuristic=None,weight='weight')
# print(path1)

# def node_to_node_cost_estimate(
#     num_orbs,
#     num_sats_per_orbs,
#     num_satellites

# ):
    
#     def heuristic_Fuc(nid_1,nid_2):
#         if nid_1>=num_satellites or nid_2>=num_satellites:
#             raise ValueError("节点编号必须小于卫星总数")
#         unit_ISLs_distance = 1
#         # 保证 node 1 的编号小于 node 2
#         nid_1,nid_2= sorted([nid_1,nid_2])
        
#         if nid_1 >= num_satellites and nid_2 >= num_satellites:
#             raise ValueError("Satellite ID must be less than the number of satellites")

#         # 计算两个卫星之间的跳数
#         node1_orbs = nid_1//num_sats_per_orbs
#         node1_id_in_orbs = nid_1 - node1_orbs*num_sats_per_orbs
#         node2_orbs = nid_2//num_sats_per_orbs
#         node2_id_in_orbs = nid_2 - node2_orbs*num_sats_per_orbs

#         intra_orbs_hop = min((node1_orbs-node2_orbs)%num_orbs, (node2_orbs-node1_orbs)%num_orbs)
#         intre_orbs_hop = min((node1_id_in_orbs-node2_id_in_orbs)%num_sats_per_orbs,(node2_id_in_orbs-node1_id_in_orbs)%num_sats_per_orbs)
#         hop_nid_1_to_nid_2 = intra_orbs_hop + intre_orbs_hop
#         print(f'orbit_1:{node1_orbs}  id_1_in_orbit:{node1_id_in_orbs}')
#         print(f'orbit_2:{node2_orbs}  id_2_in_orbit:{node2_id_in_orbs}')
#         return unit_ISLs_distance*hop_nid_1_to_nid_2
    
#     return heuristic_Fuc

# work = node_to_node_cost_estimate(72,18,72*18)
# print(0,3,work(0,3))
# print(0,11,work(0,11))
# print(0,17,work(0,17))
# print(0,26,work(0,26))
# print(0,1293,work(0,1293))