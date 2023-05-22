import math
import networkx as nx

# 输入： 地面站id、地面站可使卫星列表
# 输出： (distance_m,sid) 最近卫星的距离和 sid
def sat_closest_to_gs(
        gid,
        ground_station_satellites_in_range_candidates
):
    possible_dst_sats = ground_station_satellites_in_range_candidates[gid]
    return min(possible_dst_sats)
    

# astar 的启发函数
# 只估算 ISL 的距离，默认没有节点故障和链路故障
# 单跳 ISL 距离视为恒定不变，为轨道内单跳 ISL 的值
def node_to_node_cost_estimate(
    num_orbs,
    num_sats_per_orbs,
    num_satellites,
    sat_net_graph_only_satellites_with_isls,
):
    
    def heuristic_Fuc(nid_1,nid_2):
        if nid_1>=num_satellites or nid_2>=num_satellites:
            raise ValueError("节点编号必须小于卫星总数")
        unit_ISLs_distance = sat_net_graph_only_satellites_with_isls.edges[(nid_1%num_sats_per_orbs, (nid_1+1)%num_sats_per_orbs)]["weight"]
        # 保证 node 1 的编号小于 node 2
        nid_1,nid_2= sorted([nid_1,nid_2])
        
        if nid_1 >= num_satellites and nid_2 >= num_satellites:
            raise ValueError("Satellite ID must be less than the number of satellites")

        # 计算两个卫星之间的跳数
        node1_orbs = nid_1//num_sats_per_orbs
        node1_id_in_orbs = nid_1 - node1_orbs*num_sats_per_orbs
        node2_orbs = nid_2//num_sats_per_orbs
        node2_id_in_orbs = nid_2 - node2_orbs*num_sats_per_orbs

        intra_orbs_hop = min((node1_orbs-node2_orbs)%num_orbs, (node2_orbs-node1_orbs)%num_orbs)
        intre_orbs_hop = min((node1_id_in_orbs-node2_id_in_orbs)%num_sats_per_orbs,(node2_id_in_orbs-node1_id_in_orbs)%num_sats_per_orbs)
        hop_nid_1_to_nid_2 = intra_orbs_hop + intre_orbs_hop
        return unit_ISLs_distance*hop_nid_1_to_nid_2
    
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
    

    # Calculate shortest path distances
    if enable_verbose_logs:
        print("  > Calculating Astar for graph without ground-station relays")

    # Forwarding state
    fstate = {}

    # 实例化 astar 的启发函数
    heuristic_Fuc = node_to_node_cost_estimate(num_orbs,num_sats_per_orbs,num_satellites,sat_net_graph_only_satellites_with_isls)

    # 基于 astar 计算路由路径
    path={}
    path_len={}
    for src in range(num_satellites):
        for dst in range(num_satellites):
            path[f'{src}-{dst}']=nx.astar_path(sat_net_graph_only_satellites_with_isls, src, dst, heuristic=heuristic_Fuc, weight='weight')
            path_len[f'{src}-{dst}'] = nx.astar_path_length(sat_net_graph_only_satellites_with_isls, src,dst, heuristic=heuristic_Fuc, weight='weight')
    print(f'time_since_epoch_ns:{time_since_epoch_ns/1e9} seconds, astar calculation is completed')
    
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

                    # 当前卫星为目标卫星，直接转发到地面，否则通过 astar 算法确定下一跳
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
                            if path_len[f'{curr}-{possible_dst_sats[1]}'] < distance_m:
                                path_m = path[f'{curr}-{possible_dst_sats[1]}']
                                distance_m = path_len[f'{curr}-{possible_dst_sats[1]}']
                        
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
                    if len(ground_station_satellites_in_range_candidates[src_gid])>0 and len(ground_station_satellites_in_range_candidates[dst_gid])>0:
                        path_m =[]
                        distance_m = math.inf
                        for possible_src_sats in ground_station_satellites_in_range_candidates[src_gid]:
                            for possible_dst_sats in ground_station_satellites_in_range_candidates[dst_gid]:
                                if path_len[f'{possible_src_sats[1]}-{possible_dst_sats[1]}'] < distance_m:
                                    path_m = path[f'{possible_src_sats[1]}-{possible_dst_sats[1]}']
                                    distance_m = path_len[f'{possible_src_sats[1]}-{possible_dst_sats[1]}']
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
