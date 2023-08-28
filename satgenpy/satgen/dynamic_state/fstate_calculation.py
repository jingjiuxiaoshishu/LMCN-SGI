import math
import networkx as nx
import time
from . import cthulhu_grid_routing_algorithm as cthulhu




# 输入： 地面站id、地面站可使卫星列表
# 输出： (distance_m,sid) 最近卫星的距离和 sid
def sat_closest_to_gs(
        gid,
        ground_station_satellites_in_range_candidates
):
    possible_dst_sats = ground_station_satellites_in_range_candidates[gid]
    return min(possible_dst_sats)
    

# 实例化卫星跳数估算函数
# 只估算 ISL 的距离，默认没有节点故障和链路故障
# 单跳 ISL 距离视为恒定不变，为轨道内单跳 ISL 的值
def node_to_node_cost_estimate(
    num_orbs,
    num_sats_per_orbs,
    num_satellites,
    shift_between_last_and_first
):
    
    def heuristic_Fuc(nid_1,nid_2):
        if nid_1>=num_satellites or nid_2>=num_satellites:
            raise ValueError("节点编号必须小于卫星总数")
        # 保证 node 1 的编号小于 node 2
        nid_1,nid_2= sorted([nid_1,nid_2])
        
        if nid_1 >= num_satellites and nid_2 >= num_satellites:
            raise ValueError("Satellite ID must be less than the number of satellites")

        # 计算两个卫星之间的跳数
        node1_orbs = nid_1//num_sats_per_orbs
        node1_id_in_orbs = nid_1 - node1_orbs*num_sats_per_orbs
        node2_orbs = nid_2//num_sats_per_orbs
        node2_id_in_orbs = nid_2 - node2_orbs*num_sats_per_orbs

        min_hop_nid_1_to_nid_2_not_cross_last_to_first = (node2_orbs-node1_orbs) +  min((node1_id_in_orbs-node2_id_in_orbs)%num_sats_per_orbs,(node2_id_in_orbs-node1_id_in_orbs)%num_sats_per_orbs)

        min_hop_nid_1_to_nid_2_cross_last_to_first = (node1_orbs-node2_orbs)% num_orbs + \
                        min((node1_id_in_orbs - shift_between_last_and_first -node2_id_in_orbs)%num_sats_per_orbs,(node2_id_in_orbs + shift_between_last_and_first -node1_id_in_orbs)%num_sats_per_orbs)
        return min(min_hop_nid_1_to_nid_2_not_cross_last_to_first,min_hop_nid_1_to_nid_2_cross_last_to_first)
    
    return heuristic_Fuc


# 改名字，并非最短路径，而是 astar 求出的可用路径
def calculate_fstate_shortest_path_without_gs_relaying(
        output_dynamic_state_dir,
        time_since_epoch_ns,
        num_orbs,
        num_sats_per_orbs,
        num_satellites,
        num_ground_stations,
        sat_net_graph_only_satellites_with_isls,
        num_isls_per_sat,
        gid_to_sat_gsl_if_idx,
        ground_station_satellites_in_range_candidates,
        sat_neighbor_to_if,
        prev_fstate,
        enable_verbose_logs
):
    # 星链 72*18，相位因子为45，对应最后一个轨道与第一个轨道之间需要 9 偏移
    shift_between_last_and_first = 9
    

    # Calculate shortest path distances
    if enable_verbose_logs:
        print("  > Calculating Astar for graph without ground-station relays")

    # Forwarding state
    fstate = {}

    # 实例化 cthulhu 网络
    print("构建 cthulhu_Grid 图")
    start = time.time()
    sat_network_routing_wiht_cthulhu_Grid = cthulhu.Sat_network_routing_wiht_cthulhu_Grid(num_orbs,num_sats_per_orbs,sat_net_graph_only_satellites_with_isls,shift_between_last_and_first)
    stop = time.time()
    print(f'构建 cthulhu_Grid 图所花费时间:{stop-start}')

    print("根据 cthulhu_Grid 图， 获得每对节点的路径，以及代价")
    cthulhu_Grid_path = {}
    cthulhu_Grid_path_len = {}
    start = time.time()
    for i in range(num_orbs*num_sats_per_orbs):
        for j in  range(num_orbs*num_sats_per_orbs):
            i_to_j_path,i_to_j__path_len = sat_network_routing_wiht_cthulhu_Grid.path_a_to_b(i,j)
            cthulhu_Grid_path[(i,j)] = i_to_j_path
            cthulhu_Grid_path_len[(i,j)] = i_to_j__path_len
    stop = time.time()
    print(f'根据 cthulhu_Grid 图获得任意节点对路由所花费时间:{stop-start}')
    print(f'根据 cthulhu_Grid 图获得单源节点对路由所花费时间:{(stop-start)/num_orbs/num_sats_per_orbs}')

    # 实例化卫星跳数估算函数
    heuristic_Fuc = node_to_node_cost_estimate(num_orbs,num_sats_per_orbs,num_satellites,shift_between_last_and_first)

    print(f'time_since_epoch_ns:{time_since_epoch_ns/1e9} seconds, cthulhu calculation is completed')
    

    # Now write state to file for complete graph
    output_filename = output_dynamic_state_dir + "/fstate_" + str(time_since_epoch_ns) + ".txt"
    if enable_verbose_logs:
        print("  > Writing forwarding state to: " + output_filename)
    with open(output_filename, "w+") as f_out:
        
        # 计算源节点为卫星，目的节点为地面站的路由表
        for curr in range (num_satellites):
            for dst_gid in range(num_ground_stations):
                dst_gs_node_id = num_satellites + dst_gid

                # 默认为目标地面站无可达卫星，即 next_hop_decision 均为 -1
                next_hop_decision = (-1, -1, -1)
                dst_sat = -1
                # 如果目标地面站有可达卫星，则计算路由表
                if len(ground_station_satellites_in_range_candidates[dst_gid])>0:
                    # 如果当前卫星与目的地面站可建立链接，则更新目标卫星为当前卫星
                    for possible_dst_sats in ground_station_satellites_in_range_candidates[dst_gid]:
                        if possible_dst_sats[1] == curr:
                            dst_sat = curr
                            break

                    # 当前卫星为目标卫星，直接转发到地面，否则通过 cthulhu_grid 确定下一跳
                    if curr == dst_sat and dst_sat!=-1:
                        next_hop_decision = (
                            dst_gs_node_id,
                            num_isls_per_sat[dst_sat] + gid_to_sat_gsl_if_idx[dst_gid],
                            0
                        )
                    else:
                        path_m =[]
                        distance_m = math.inf
                        for possible_dst_sats in ground_station_satellites_in_range_candidates[dst_gid]:
                            if heuristic_Fuc(curr,possible_dst_sats[1]) < distance_m:
                                path_m = cthulhu_Grid_path[(curr,possible_dst_sats[1])]
                                distance_m = heuristic_Fuc(curr,possible_dst_sats[1])
                        
                        if not math.isinf(distance_m):
                            next_hop_decision = (
                                path_m[1],
                                sat_neighbor_to_if[(curr, path_m[1])],
                                sat_neighbor_to_if[(path_m[1], curr)]
                            )
                
                # Write to forwarding state
                if not prev_fstate or prev_fstate[(curr, dst_gs_node_id)] != next_hop_decision:
                    f_out.write("%d,%d,%d,%d,%d\n" % (
                        curr,
                        dst_gs_node_id,
                        next_hop_decision[0],
                        next_hop_decision[1],
                        next_hop_decision[2]
                    ))
                fstate[(curr, dst_gs_node_id)] = next_hop_decision

        for src_gid in range (num_ground_stations):
            for dst_gid in range(num_ground_stations):
                if src_gid != dst_gid:
                    src_gs_node_id = num_satellites + src_gid
                    dst_gs_node_id = num_satellites + dst_gid
                    # 默认为源地面站无可接入卫星，或目标地面站无可达卫星，即 next_hop_decision 均为 -1
                    next_hop_decision = (-1, -1, -1)

                    # 若源地面站与目的地面站均有可达卫星
                    if len(ground_station_satellites_in_range_candidates[src_gid])>0 and len(ground_station_satellites_in_range_candidates[dst_gid])>0:
                        path_m =[]
                        distance_m = math.inf
                        for possible_src_sats in ground_station_satellites_in_range_candidates[src_gid]:
                            for possible_dst_sats in ground_station_satellites_in_range_candidates[dst_gid]:
                                if heuristic_Fuc(possible_src_sats[1],possible_dst_sats[1] )< distance_m:
                                    path_m = cthulhu_Grid_path[(possible_src_sats[1],possible_dst_sats[1] )]
                                    distance_m = heuristic_Fuc(possible_src_sats[1],possible_dst_sats[1] )
                        if not math.isinf(distance_m):
                            next_hop_decision = (
                                path_m[0],
                                0,
                                num_isls_per_sat[path_m[0]] + gid_to_sat_gsl_if_idx[src_gid]
                            )

                    # Write to forwarding state
                    if not prev_fstate or prev_fstate[(src_gs_node_id, dst_gs_node_id)] != next_hop_decision:
                        f_out.write("%d,%d,%d,%d,%d\n" % (
                            src_gs_node_id,
                            dst_gs_node_id,
                            next_hop_decision[0],
                            next_hop_decision[1],
                            next_hop_decision[2]
                        ))
                    fstate[(src_gs_node_id, dst_gs_node_id)] = next_hop_decision
    # Finally return result
    return fstate


def calculate_fstate_shortest_path_with_gs_relaying(
        output_dynamic_state_dir,
        time_since_epoch_ns,
        num_satellites,
        num_ground_stations,
        sat_net_graph,
        num_isls_per_sat,
        gid_to_sat_gsl_if_idx,
        sat_neighbor_to_if,
        prev_fstate,
        enable_verbose_logs
):

    # Calculate shortest paths
    if enable_verbose_logs:
        print("  > Calculating Floyd-Warshall for graph including ground-station relays")
    # (Note: Numpy has a deprecation warning here because of how networkx uses matrices)
    dist_sat_net = nx.floyd_warshall_numpy(sat_net_graph)

    # Forwarding state
    fstate = {}

    # Now write state to file for complete graph
    output_filename = output_dynamic_state_dir + "/fstate_" + str(time_since_epoch_ns) + ".txt"
    if enable_verbose_logs:
        print("  > Writing forwarding state to: " + output_filename)
    with open(output_filename, "w+") as f_out:

        # Satellites and ground stations to ground stations
        for current_node_id in range(num_satellites + num_ground_stations):
            for dst_gid in range(num_ground_stations):
                dst_gs_node_id = num_satellites + dst_gid

                # Cannot forward to itself
                if current_node_id != dst_gs_node_id:

                    # Among its neighbors, find the one which promises the
                    # lowest distance to reach the destination satellite
                    next_hop_decision = (-1, -1, -1)
                    best_distance_m = 1000000000000000
                    for neighbor_id in sat_net_graph.neighbors(current_node_id):

                        # Any neighbor must be reachable
                        if math.isinf(dist_sat_net[(current_node_id, neighbor_id)]):
                            raise ValueError("Neighbor cannot be unreachable")

                        # Calculate distance = next-hop + distance the next hop node promises
                        distance_m = (
                            sat_net_graph.edges[(current_node_id, neighbor_id)]["weight"]
                            +
                            dist_sat_net[(neighbor_id, dst_gs_node_id)]
                        )
                        if (
                                not math.isinf(dist_sat_net[(neighbor_id, dst_gs_node_id)])
                                and
                                distance_m < best_distance_m
                        ):

                            # Check node identifiers to determine what are the
                            # correct interface identifiers
                            if current_node_id >= num_satellites and neighbor_id < num_satellites:  # GS to sat.
                                my_if = 0
                                next_hop_if = (
                                    num_isls_per_sat[neighbor_id]
                                    +
                                    gid_to_sat_gsl_if_idx[current_node_id - num_satellites]
                                )

                            elif current_node_id < num_satellites and neighbor_id >= num_satellites:  # Sat. to GS
                                my_if = (
                                    num_isls_per_sat[current_node_id]
                                    +
                                    gid_to_sat_gsl_if_idx[neighbor_id - num_satellites]
                                )
                                next_hop_if = 0

                            elif current_node_id < num_satellites and neighbor_id < num_satellites:  # Sat. to sat.
                                my_if = sat_neighbor_to_if[(current_node_id, neighbor_id)]
                                next_hop_if = sat_neighbor_to_if[(neighbor_id, current_node_id)]

                            else:  # GS to GS
                                raise ValueError("GS-to-GS link cannot exist")

                            # Write the next-hop decision
                            next_hop_decision = (
                                neighbor_id,  # Next-hop node identifier
                                my_if,        # My outgoing interface id
                                next_hop_if   # Next-hop incoming interface id
                            )

                            # Update best distance found
                            best_distance_m = distance_m

                    # Write to forwarding state
                    if not prev_fstate or prev_fstate[(current_node_id, dst_gs_node_id)] != next_hop_decision:
                        f_out.write("%d,%d,%d,%d,%d\n" % (
                            current_node_id,
                            dst_gs_node_id,
                            next_hop_decision[0],
                            next_hop_decision[1],
                            next_hop_decision[2]
                        ))
                    fstate[(current_node_id, dst_gs_node_id)] = next_hop_decision

    # Finally return result
    return fstate
