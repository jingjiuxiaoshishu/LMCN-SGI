import math
import networkx as nx
import ephem
from astropy import units as u


def get_visible_left_of_gsl(sid,gid,now,visible_times):
    if now >= visible_times[gid][sid][0] and now <= visible_times[gid][sid][1]:
        visible_time_left = (visible_times[gid][sid][1] - now)/ephem.second
        # print(visible_time_left,gid,sid)

        return visible_time_left
    else:
        return 0



def calculate_fstate_shortest_path_without_gs_relaying(
        output_dynamic_state_dir,
        epoch,
        time_since_epoch_ns,
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
    
    # 将距离转化为跳数
    for src,dst,attrs in sat_net_graph_only_satellites_with_isls.edges(data=True):
        edge = (src,dst)
        edge_weight = attrs["weight"]
        if edge_weight != math.inf:
            sat_net_graph_only_satellites_with_isls[edge[0]][edge[1]]["weight"] = 1

    # 计算最短跳数及路径
    d_path_without_gs = dict(nx.all_pairs_dijkstra_path(sat_net_graph_only_satellites_with_isls))
    d_path_len_without_gs =  dict(nx.all_pairs_dijkstra_path_length(sat_net_graph_only_satellites_with_isls))

    time = epoch + time_since_epoch_ns * u.ns

    
    # Forwarding state
    fstate = {}
    
    # 路由状态写入目录
    output_filename = output_dynamic_state_dir + "/fstate_" + str(time_since_epoch_ns) + ".txt"
    if enable_verbose_logs:
        print("  > Writing forwarding state to: " + output_filename)

    import pickle
    with open("visible_times.pkl","rb") as f:
        visible_times = pickle.load(f)

    # 提前地面站本时隙的可见卫星，以便取用
    # gid : list of sid
    sats_visible_of_gs = []
    for gid,sats_visible_of_gs_with_range in enumerate(ground_station_satellites_in_range_candidates):
        sats_visible_of_gs.append([])
        for distance,sid in sats_visible_of_gs_with_range:
            if distance != math.inf:
                sats_visible_of_gs[gid].append(sid)
        
        # print(gid,":",sats_visible_of_gs[gid])

    

    # 最小跳且最大服务时长的 waypoint
    waypoint_with_minimum_hop_and_maximum_service_time = {}
    for sid in range(num_satellites):
        waypoint_with_minimum_hop_and_maximum_service_time[sid] = {}    
        for gid in range(num_ground_stations):
            gs_node_id = gid + num_satellites
            waypoint_sid = -1
            hop_to_waypoint = math.inf
            service_time_of_waypoint = 0
            for sat_visible in sats_visible_of_gs[gid]:
                hop = d_path_len_without_gs[sid][sat_visible]
                visible_time   = get_visible_left_of_gsl(sat_visible,gid,ephem.Date(time.datetime),visible_times)
                if hop < hop_to_waypoint  or (hop == hop_to_waypoint and visible_time>service_time_of_waypoint):
                    if visible_time>0:
                        waypoint_sid = sat_visible
                        hop_to_waypoint = hop
                        service_time_of_waypoin = visible_time
                        # print("lalalal")
            waypoint_with_minimum_hop_and_maximum_service_time[sid][gs_node_id] = waypoint_sid
            # if waypoint_sid!=-1:
            #     print(sid,gid,waypoint_sid)



    with open(output_filename, "w+") as f_out:
        for curr in range(num_satellites):
            for dst_gid in range(num_ground_stations):
                next_hop_decision = (-1, -1, -1, -1, -1)
                dst_gs_node_id = dst_gid + num_satellites
                if prev_fstate:
                    (
                        pre_next_hop,
                        curr_if,
                        next_hop_if,
                        pre_sat_src_gs_link,
                        pre_sat_dst_gs_link
                    ) = prev_fstate[(curr, dst_gs_node_id)]
                    if pre_next_hop != -1 and d_path_len_without_gs[curr][pre_sat_dst_gs_link] != math.inf and (pre_sat_dst_gs_link in sats_visible_of_gs[dst_gid]):
                        if curr == pre_sat_dst_gs_link:
                            next_hop_decision = (
                                dst_gs_node_id,
                                num_isls_per_sat[curr] + gid_to_sat_gsl_if_idx[dst_gid],
                                0,
                                curr,
                                curr
                            )
                        else:
                            next_hop = d_path_without_gs[curr][pre_sat_dst_gs_link][1]
                            next_hop_decision = (
                                next_hop,
                                sat_neighbor_to_if[(curr, next_hop)],
                                sat_neighbor_to_if[(next_hop, curr)],
                                curr,
                                pre_sat_dst_gs_link
                            )

                # 如果需要更新，进入更新流程
                if next_hop_decision == (-1, -1, -1, -1, -1):
                    next_hop = -1
                    waypoint = waypoint_with_minimum_hop_and_maximum_service_time[curr][dst_gs_node_id]
                    if waypoint !=-1 and d_path_len_without_gs[curr][waypoint]!=math.inf:
                        if waypoint == curr:
                            next_hop = dst_gs_node_id
                            next_hop_decision = (
                                dst_gs_node_id,
                                num_isls_per_sat[curr] + gid_to_sat_gsl_if_idx[dst_gid],
                                0,
                                curr,
                                curr
                            )
                        else:
                            next_hop = d_path_without_gs[curr][waypoint][1]
                            next_hop_decision = (
                            next_hop,
                            sat_neighbor_to_if[(curr, next_hop)],
                            sat_neighbor_to_if[(next_hop, curr)],
                            curr,
                            waypoint
                        )
                
                # Update forwarding state
                if not prev_fstate or prev_fstate[(curr, dst_gs_node_id)] != next_hop_decision:
                    f_out.write("%d,%d,%d,%d,%d\n" % (
                        curr,
                        dst_gs_node_id,
                        next_hop_decision[0],
                        next_hop_decision[1],
                        next_hop_decision[2]
                    ))
                fstate[(curr, dst_gs_node_id)] = next_hop_decision

        # 重置 waypoint 列表
        # waypoint_with_minimum_hop_and_maximum_service_time = {}
        for src_gid in range(num_ground_stations):
            src_gs_node_id = src_gid + num_satellites
            waypoint_with_minimum_hop_and_maximum_service_time[src_gs_node_id] = {}
            for dst_gid in range(num_ground_stations):
                if src_gid != dst_gid:
                    next_hop_decision = (-1, -1, -1, -1, -1)
                    dst_gs_node_id = dst_gid + num_satellites

                    if prev_fstate:
                        (
                            pre_next_hop,
                            curr_if,
                            next_hop_if,
                            pre_sat_src_gs_link,
                            pre_sat_dst_gs_link
                        ) = prev_fstate[(src_gs_node_id, dst_gs_node_id)]

                        # 注意更新的时候使用的是 fstate 而非 pre_fstate ，以当前为准
                        if pre_next_hop != -1 and pre_next_hop in sats_visible_of_gs[src_gid] and fstate[(pre_next_hop, dst_gs_node_id)][0]!= -1:
                            # print("pre_sat_src_gs_link_avaiable and pre_sat_dst_gs_link_avaiable\n\n")
                            next_hop_decision = (
                                pre_next_hop,
                                0,
                                num_isls_per_sat[pre_next_hop] + gid_to_sat_gsl_if_idx[src_gid],
                                pre_next_hop,
                                fstate[(pre_next_hop, dst_gs_node_id)][4]
                            )
                    
                    # 是否需要重新计算路由
                    if next_hop_decision == (-1, -1, -1, -1, -1):
                        waypoint_pairs = -1
                        hop_between_waypoint_pairs = math.inf
                        service_time_between_waypoint_pairs = 0
                        # 查看源地面站的可见卫星
                        for sat_visible in sats_visible_of_gs[src_gid]:
                            link_exist = fstate[(sat_visible, dst_gs_node_id)][0]
                            if link_exist!=-1:
                                candidate_waypoint_pairs = [sat_visible,fstate[(sat_visible, dst_gs_node_id)][4]]
                                service_time = min(get_visible_left_of_gsl(candidate_waypoint_pairs[0],src_gid,ephem.Date(time.datetime),visible_times),get_visible_left_of_gsl(candidate_waypoint_pairs[1],dst_gid,ephem.Date(time.datetime),visible_times))
                                hop = d_path_len_without_gs[candidate_waypoint_pairs[0]][candidate_waypoint_pairs[1]]
                                if hop<hop_between_waypoint_pairs or (hop == hop_between_waypoint_pairs and service_time<service_time_between_waypoint_pairs):
                                    waypoint_pairs = candidate_waypoint_pairs
                                    hop_between_waypoint_pairs = hop
                                    service_time_between_waypoint_pairs = service_time
                        if waypoint_pairs != -1:
                            next_hop_decision = (
                                waypoint_pairs[0],
                                0,
                                num_isls_per_sat[waypoint_pairs[0]] + gid_to_sat_gsl_if_idx[src_gid],
                                waypoint_pairs[0],
                                waypoint_pairs[1]
                            )
                    
                    # Update forwarding state
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
