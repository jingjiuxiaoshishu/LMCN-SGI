import math
import networkx as nx


def calculate_fstate_shortest_path_without_gs_relaying(
        output_dynamic_state_dir,
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

    # Calculate shortest path distances
    if enable_verbose_logs:
        print("  > Calculating Floyd-Warshall for graph without ground-station relays")
    # (Note: Numpy has a deprecation warning here because of how networkx uses matrices)
    # dist_sat_net_without_gs = nx.floyd_warshall_numpy(sat_net_graph_only_satellites_with_isls)
    d_path_without_gs = dict(nx.all_pairs_dijkstra_path(sat_net_graph_only_satellites_with_isls))
    d_path_len_without_gs =  dict(nx.all_pairs_dijkstra_path_length(sat_net_graph_only_satellites_with_isls))
    # Forwarding state
    fstate = {}

    # Now write state to file for complete graph
    output_filename = output_dynamic_state_dir + "/fstate_" + str(time_since_epoch_ns) + ".txt"
    if enable_verbose_logs:
        print("  > Writing forwarding state to: " + output_filename)
    with open(output_filename, "w+") as f_out:
        dist_satellite_to_ground_station = {}
        for curr in range(num_satellites):
            for dst_gid in range(num_ground_stations):  

                dst_gs_node_id = num_satellites + dst_gid
                dist_satellite_to_ground_station[(curr,dst_gs_node_id)] = math.inf

                possible_dst_sats = ground_station_satellites_in_range_candidates[dst_gid]
                # next_hop,if_curr,if_next,curr,visible_sat_sector
                next_hop_decision = (-1, -1, -1, -1, -1)

                if prev_fstate :
                    pre_sat_gs_link =  prev_fstate[(curr, dst_gs_node_id)][4]
                    if pre_sat_gs_link != -1 and d_path_len_without_gs[curr][pre_sat_gs_link] != math.inf:
                        for distance_to_possible_dst_sat,possible_dst_sat in possible_dst_sats:
                            if possible_dst_sat == pre_sat_gs_link:
                                if pre_sat_gs_link == curr:
                                    print("pre_sat_gs_link == curr\n\n",curr)
                                    next_hop_decision = (
                                        dst_gs_node_id,
                                        num_isls_per_sat[curr] + gid_to_sat_gsl_if_idx[dst_gid],
                                        0,
                                        curr,
                                        curr
                                    )
                                    dist_satellite_to_ground_station[(curr,dst_gs_node_id)]  = distance_to_possible_dst_sat
                                    break
                                else:
                                    # print("pre_sat_gs_link != curr\n\n")
                                    next_hop = d_path_without_gs[curr][pre_sat_gs_link][1]
                                    next_hop_decision = (
                                        next_hop,
                                        sat_neighbor_to_if[(curr, next_hop)],
                                        sat_neighbor_to_if[(next_hop, curr)],
                                        curr,
                                        pre_sat_gs_link
                                    )
                                    dist_satellite_to_ground_station[(curr,dst_gs_node_id)]  = distance_to_possible_dst_sat + d_path_len_without_gs[curr][pre_sat_gs_link]
                                    break
                # 如果在上一步没有能更新，即上一条选中 gsl 已经失效或者无法到达该 gsl 链接的卫星，需要进入路由更新
                # 或者 prev_fstate 为空，即仿真的第 0 个时隙
                if next_hop_decision == (-1, -1, -1, -1, -1):
                    best_distance_m = 1000000000000000
                    for distance_to_possible_dst_sat,possible_dst_sat in possible_dst_sats:
                        # 如果 curr 对该地面站可见
                        if  d_path_len_without_gs[curr][possible_dst_sat] == 0:
                            best_distance_m = distance_to_possible_dst_sat
                            next_hop_decision = (
                                    dst_gs_node_id,
                                    num_isls_per_sat[curr] + gid_to_sat_gsl_if_idx[dst_gid],
                                    0,
                                    curr,
                                    curr
                            )
                            dist_satellite_to_ground_station[(curr,dst_gs_node_id)]  = distance_to_possible_dst_sat
                            # 注意这里要添加 break
                            break
                        # curr 不可见时
                        elif d_path_len_without_gs[curr][possible_dst_sat] != math.inf:
                            if d_path_len_without_gs[curr][possible_dst_sat] + distance_to_possible_dst_sat < best_distance_m:
                                best_distance_m = d_path_len_without_gs[curr][possible_dst_sat] + distance_to_possible_dst_sat
                                next_hop = d_path_without_gs[curr][possible_dst_sat][1]
                                next_hop_decision = (
                                    next_hop,
                                    sat_neighbor_to_if[(curr, next_hop)],
                                    sat_neighbor_to_if[(next_hop, curr)],
                                    curr,
                                    possible_dst_sat
                                )
                                dist_satellite_to_ground_station[(curr,dst_gs_node_id)]  = distance_to_possible_dst_sat +  d_path_len_without_gs[curr][possible_dst_sat]
                # if next_hop_decision == (-1, -1, -1, -1, -1):
                #     print("what the fuck!!!\n")

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
        
        for src_gid in range(num_ground_stations):
            for dst_gid in range(num_ground_stations):
                if src_gid != dst_gid:
                    src_gs_node_id = num_satellites + src_gid
                    dst_gs_node_id = num_satellites + dst_gid

                    next_hop_decision = (-1, -1, -1, -1, -1)

                    possible_src_sats = ground_station_satellites_in_range_candidates[src_gid]
                    possible_dst_sats = ground_station_satellites_in_range_candidates[dst_gid]

                    if prev_fstate :
                        pre_sat_src_gs_link =  prev_fstate[(src_gs_node_id, dst_gs_node_id)][3]
                        pre_sat_dst_gs_link =  prev_fstate[(src_gs_node_id, dst_gs_node_id)][4]

                        if pre_sat_src_gs_link != -1 and pre_sat_dst_gs_link != -1 and d_path_len_without_gs[pre_sat_src_gs_link][pre_sat_dst_gs_link] != math.inf:
                            pre_sat_src_gs_link_avaiable = False
                            pre_sat_dst_gs_link_avaiable = False
                            for distance_to_possible_src_sat,possible_src_sat in possible_src_sats:
                                if possible_src_sat == pre_sat_src_gs_link:
                                    pre_sat_src_gs_link_avaiable = True
                            for distance_to_possible_dst_sat,possible_dst_sat in possible_dst_sats:
                                if possible_dst_sat == pre_sat_dst_gs_link:
                                    pre_sat_dst_gs_link_avaiable = True
                            if pre_sat_src_gs_link_avaiable and pre_sat_dst_gs_link_avaiable:
                                # print("pre_sat_src_gs_link_avaiable and pre_sat_dst_gs_link_avaiable\n\n")
                                next_hop_decision = (
                                    pre_sat_src_gs_link,
                                    0,
                                    num_isls_per_sat[pre_sat_src_gs_link] + gid_to_sat_gsl_if_idx[src_gid],
                                    pre_sat_src_gs_link,
                                    pre_sat_dst_gs_link
                                )
                    # 如果前一步没有能更新，即原本使用的 gsl 至少一条失效，或者两颗gsl连接的卫星之间无路径
                    if next_hop_decision == (-1, -1, -1, -1, -1):
                        best_distance_m = 1000000000000000
                        for distance_to_possible_src_sat,possible_src_sat in possible_src_sats:
                            # 源地面站的某可见卫星到目的地面站有路径
                            if fstate[(possible_src_sat, dst_gs_node_id)][2] != -1:
                                if distance_to_possible_src_sat + dist_satellite_to_ground_station[(possible_src_sat,dst_gs_node_id)] < best_distance_m:
                                    best_distance_m = distance_to_possible_src_sat + dist_satellite_to_ground_station[(possible_src_sat,dst_gs_node_id)]
                                    next_hop_decision = (
                                        possible_src_sat,
                                        0,
                                        num_isls_per_sat[possible_src_sat] + gid_to_sat_gsl_if_idx[src_gid],
                                        possible_src_sat,
                                        fstate[(possible_src_sat, dst_gs_node_id)][4]
                                    )
                    # if next_hop_decision == (-1, -1, -1, -1, -1):
                    #     print("rnm tuiqian\n\n")
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
