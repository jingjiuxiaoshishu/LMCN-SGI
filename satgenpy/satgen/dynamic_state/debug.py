import networkx as nx

# 创建一个有向图
G = nx.DiGraph()

# 添加节点和边
for int 
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

# 定义起点和终点
start = 'A'
end = 'A'

# 使用 A* 算法寻找最短路径
path = nx.astar_path(G, start, end, heuristic=None, weight='weight')
path_len = nx.astar_path_length(G, start, end, heuristic=None, weight='weight')

# 打印路径
print(" -> ".join(path))
print(len(path))
print(path_len)
