# The MIT License (MIT)
#
# Copyright (c) 2020 ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from satgen.distance_tools import *
from astropy import units as u
import ephem
import math
import networkx as nx
import numpy as np
from .algorithm_free_one_only_gs_relays import algorithm_free_one_only_gs_relays
from .algorithm_free_one_only_over_isls import algorithm_free_one_only_over_isls
from .algorithm_paired_many_only_over_isls import algorithm_paired_many_only_over_isls
from .algorithm_free_gs_one_sat_many_only_over_isls import algorithm_free_gs_one_sat_many_only_over_isls
from .distrubute_route.satellite_node import Satellite_node
from .distrubute_route.visible_helper import Visible_time_helper
from .distrubute_route.olsr_node import Olsr_node
from .distrubute_route.sat_selector import Sat_selector
import copy

def get_gsl_list(satellites,ground_stations,max_gsl_length_m,epoch,time):
    gsl_list = {}
    for gid,ground_station in enumerate(ground_stations):
        gsl_list[gid] = []
        for sid in range(len(satellites)):
            distance_m = distance_m_ground_station_to_satellite(
                ground_station,
                satellites[sid],
                str(epoch),
                str(time)
            )
            if distance_m <= max_gsl_length_m:
                gsl_list[gid].append(sid)
    return gsl_list


def check_gsl(sid,gid,satellites,ground_stations,max_gsl_length_m,epoch,time):
    distance_m = distance_m_ground_station_to_satellite(
        ground_stations[gid],
        satellites[sid],
        str(epoch),
        str(time)
    )
    if distance_m <= max_gsl_length_m:
        return True
    else:
        return False

def get_sat_net_graph_with_sat_and_gs(satellites,ground_stations,sat_net_graph_only_satellites_with_isls,max_gsl_length_m,epoch,time):
    print("构建星地完整拓扑图")
    sat_net_graph_with_sat_and_gs = copy.deepcopy(sat_net_graph_only_satellites_with_isls)
    for gid in range(len(ground_stations)):
        sat_net_graph_with_sat_and_gs.add_edge(gid+len(satellites),0,weight=math.inf)
    gsl_list = get_gsl_list(satellites,ground_stations,max_gsl_length_m,epoch,time)
    for gs_node_id,sid in gsl_list:
        distance_m = distance_m_ground_station_to_satellite(
                ground_stations[gs_node_id-len(satellites)],
                satellites[sid],
                str(epoch),
                str(time)
            )
        sat_net_graph_with_sat_and_gs.add_edge(gs_node_id,sid,weight = distance_m)
    return sat_net_graph_with_sat_and_gs
    





def generate_dynamic_state(
        output_dynamic_state_dir,
        epoch,
        simulation_end_time_ns,
        time_step_ns,
        offset_ns,
        num_orbs,
        num_sats_per_orbs,
        satellites,
        ground_stations,
        list_isls,
        list_gsl_interfaces_info,
        max_gsl_length_m,
        max_isl_length_m,
        dynamic_state_algorithm,  # Options:
                                  # "algorithm_free_one_only_gs_relays"
                                  # "algorithm_free_one_only_over_isls"
                                  # "algorithm_paired_many_only_over_isls"
        enable_verbose_logs
):
    if offset_ns % time_step_ns != 0:
        raise ValueError("Offset must be a multiple of time_step_ns")
    
    # 建立 有距离的星上拓扑图 和 边恒为一的星上拓扑图
    sat_net_graph_only_satellites_with_isls = nx.Graph()

    for sid_1,sid_2 in list_isls:
        sat_distance_m = distance_m_between_satellites(satellites[sid_1], satellites[sid_2], str(epoch), str(epoch))
        sat_net_graph_only_satellites_with_isls.add_edge(sid_1,sid_2,weight = sat_distance_m)
  

    time_step_ms = time_step_ns/1000/1000
    simulation_end_time_s = simulation_end_time_ns/1000/1000/1000

    print("构建 olsr 节点")
    olsr_nodes = []
    for node_id in range(len(satellites)):
        olsr_node = Olsr_node(node_id,len(satellites), sat_net_graph_only_satellites_with_isls, epoch, time_step_ms, simulation_end_time_s)
        olsr_nodes.append(olsr_node)

    for olsr_node in olsr_nodes:
        olsr_node.update_point_to_all_node(olsr_nodes)


    prev_output = None
    i = 0
    total_iterations = ((simulation_end_time_ns - offset_ns) / time_step_ns)
    for time_since_epoch_ns in range(offset_ns, simulation_end_time_ns, time_step_ns):
        if not enable_verbose_logs:
            if i % int(math.floor(total_iterations) / 10.0) == 0:
                print("Progress: calculating for T=%d (time step granularity is still %d ms)" % (
                    time_since_epoch_ns, time_step_ns / 1000000
                ))
            i += 1

        
        # sat_net_graph_only_satellites_with_isls 更新
        print("\n sat_net_graph_only_satellites_with_isls 更新")
        if time_since_epoch_ns/time_step_ns == 1:
            import random
            import pickle
            # 设置随机种子
            random.seed(42)
            # 百分之一的坏边
            print("\n 随机构建坏边，坏边率默认为百分之 1")
            percentage = 0.01
            num_fail_edges = int(percentage * len(list_isls))
            fail_edges = set(random.sample(list_isls, num_fail_edges))
            with open("fail_edges.pkl","wb") as f:
                pickle.dump(fail_edges,f) 
            for fail_edge in fail_edges:
                sat_net_graph_only_satellites_with_isls[fail_edge[0]][fail_edge[1]]["weight"] = math.inf
        time = epoch + time_since_epoch_ns * u.ns
        for src,dst,attrs in sat_net_graph_only_satellites_with_isls.edges(data=True):
            edge = (src,dst)
            edge_weight = attrs["weight"]
            if edge_weight != math.inf:
                distance = distance_m_between_satellites(satellites[edge[0]], satellites[edge[1]], str(epoch), str(time))
                sat_net_graph_only_satellites_with_isls[edge[0]][edge[1]]["weight"] = distance

        print("\n 更新 olsr 节点维护的图，并 process、\n")
        for olsr_node in olsr_nodes:
            # 待修改名字
            olsr_node.update_sat_net_graph_with_sat_and_gs(sat_net_graph_only_satellites_with_isls)
            olsr_node.process()
        # print("check olsr_node 0 的路由表",olsr_nodes[0].forwarding_table)

        prev_output = generate_dynamic_state_distribute(
            epoch,
            olsr_nodes,
            satellites,
            ground_stations,
            output_dynamic_state_dir,
            time_since_epoch_ns,
            list_isls,
            max_isl_length_m,
            max_gsl_length_m,
            sat_net_graph_only_satellites_with_isls,
            list_gsl_interfaces_info,
            prev_output,
            enable_verbose_logs
        )

    overhead = 0 
    for olsr_node in olsr_nodes:
        overhead = overhead + olsr_node.num_msg_rev

    with open("overhead.txt","a+") as f:
        f.write(f"overhead:{overhead} overhead:{percentage} olsr 仿真{simulation_end_time_s}s")



    

def generate_dynamic_state_distribute(
        epoch,
        olsr_nodes,
        satellites,
        ground_stations,
        output_dynamic_state_dir,
        time_since_epoch_ns,
        list_isls,
        max_isl_length_m,
        max_gsl_length_m,
        sat_net_graph_only_satellites_with_isls,
        list_gsl_interfaces_info,
        prev_output,
        enable_verbose_logs
):
    
    prev_fstate = None
    if prev_output is not None:
        prev_fstate = prev_output["fstate"]
    
    #################################
    if enable_verbose_logs:
        print("\nISL INFORMATION")

    # ISL edges
    total_num_isls = 0
    num_isls_per_sat = [0] * len(satellites)
    sat_neighbor_to_if = {}
    for (a, b) in list_isls:

        # ISLs are not permitted to exceed their maximum distance
        # TODO: Technically, they can (could just be ignored by forwarding state calculation),
        # TODO: but practically, defining a permanent ISL between two satellites which
        # TODO: can go out of distance is generally unwanted
        sat_distance_m = distance_m_between_satellites(satellites[a], satellites[b], str(epoch), str(epoch + time_since_epoch_ns*u.ns))
        if sat_distance_m > max_isl_length_m:
            raise ValueError(
                "The distance between two satellites (%d and %d) "
                "with an ISL exceeded the maximum ISL length (%.2fm > %.2fm at t=%dns)"
                % (a, b, sat_distance_m, max_isl_length_m, time_since_epoch_ns)
            )

        # 修改状态正常的边的 weight
        if sat_net_graph_only_satellites_with_isls[a][b]["weight"] != math.inf:
            sat_net_graph_only_satellites_with_isls[a][b]["weight"] = sat_distance_m


        # Interface mapping of ISLs
        sat_neighbor_to_if[(a, b)] = num_isls_per_sat[a]
        sat_neighbor_to_if[(b, a)] = num_isls_per_sat[b]
        num_isls_per_sat[a] += 1
        num_isls_per_sat[b] += 1
        total_num_isls += 1

    if enable_verbose_logs:
        print("  > Total ISLs............. " + str(len(list_isls)))
        print("  > Min. ISLs/satellite.... " + str(np.min(num_isls_per_sat)))
        print("  > Max. ISLs/satellite.... " + str(np.max(num_isls_per_sat)))

    #################################

    if enable_verbose_logs:
        print("\nGSL INTERFACE INFORMATION")

    satellite_gsl_if_count_list = list(map(
        lambda x: x["number_of_interfaces"],
        list_gsl_interfaces_info[0:len(satellites)]
    ))
    ground_station_gsl_if_count_list = list(map(
        lambda x: x["number_of_interfaces"],
        list_gsl_interfaces_info[len(satellites):(len(satellites)+len(ground_stations))]
    ))
    if enable_verbose_logs:
        print("  > Min. GSL IFs/satellite........ " + str(np.min(satellite_gsl_if_count_list)))
        print("  > Max. GSL IFs/satellite........ " + str(np.max(satellite_gsl_if_count_list)))
        print("  > Min. GSL IFs/ground station... " + str(np.min(ground_station_gsl_if_count_list)))
        print("  > Max. GSL IFs/ground_station... " + str(np.max(ground_station_gsl_if_count_list)))



    print("写入 gsl 接口的带宽 gsl_if_bandwidth_")
    output_filename = output_dynamic_state_dir + "/gsl_if_bandwidth_" + str(time_since_epoch_ns) + ".txt"
    if enable_verbose_logs:
        print("  > Writing interface bandwidth state to: " + output_filename)
    with open(output_filename, "w+") as f_out:
        if time_since_epoch_ns == 0:
            for node_id in range(len(satellites)):
                f_out.write("%d,%d,%f\n"
                            % (node_id, num_isls_per_sat[node_id],
                               list_gsl_interfaces_info[node_id]["aggregate_max_bandwidth"]))
            for node_id in range(len(satellites), (len(satellites)+len(ground_stations))):
                f_out.write("%d,%d,%f\n"
                            % (node_id, 0, list_gsl_interfaces_info[node_id]["aggregate_max_bandwidth"]))


    #################################
    if enable_verbose_logs:
        print(f"\n 更新 {str(epoch + time_since_epoch_ns*u.ns)} 时刻路由表")
    
    time = epoch + time_since_epoch_ns * u.ns

    gsl_list = get_gsl_list(satellites,ground_stations,max_gsl_length_m,epoch,time)


    forward_table_sat_to_gs = {}
    forward_cost_sat_to_gs = {}
    for sid in range(len(satellites)):
        forward_table_sat_to_gs[sid] = {}
        forward_cost_sat_to_gs[sid] = {}

        for gid in range(len(ground_stations)):
            if check_gsl(sid,gid,satellites,ground_stations,max_gsl_length_m,epoch,time):
                forward_table_sat_to_gs[sid][gid] = gid+len(satellites)
                forward_cost_sat_to_gs[sid][gid] = 1
            else:
                next_hop = -1
                cost = math.inf
                for waypoint_sid in gsl_list[gid]:
                    if olsr_nodes[sid].forwarding_cost[waypoint_sid] < cost:
                        next_hop = olsr_nodes[sid].forwarding_table[waypoint_sid]
                        cost = olsr_nodes[sid].forwarding_cost[waypoint_sid]
                forward_table_sat_to_gs[sid][gid] = next_hop
                forward_cost_sat_to_gs[sid][gid] = cost+1
    

    # 初始化一个空的 gs 到 gs 的路由表
    forward_table_gs_to_gs = {}
    forward_cost_gs_to_gs = {}
    for gid_1 in range(len(ground_stations)):
        forward_table_gs_to_gs[gid_1] = {}
        forward_cost_gs_to_gs[gid_1] = {}
        for gid_2 in range(len(ground_stations)):
            if gid_1 != gid_2:
                next_hop = -1
                cost = math.inf
                for waypoint_sid in gsl_list[gid_1]:
                    if forward_cost_sat_to_gs[waypoint_sid][gid_2] < cost:
                        next_hop = waypoint_sid
                        cost = forward_cost_sat_to_gs[waypoint_sid][gid_2]
                cost = cost + 1

                forward_table_gs_to_gs[gid_1][gid_2] = next_hop
                forward_cost_gs_to_gs[gid_1][gid_2] = cost



    # Forwarding state
    fstate = {}
    num_gs = len(ground_stations)
    gid_to_sat_gsl_if_idx = [0] * num_gs

    # Now write state to file for complete graph
    output_filename = output_dynamic_state_dir + "/fstate_" + str(time_since_epoch_ns) + ".txt"
    if enable_verbose_logs:
        print("  > Writing forwarding state to: " + output_filename)
    with open(output_filename, "w+") as f_out:
        for curr_sid,_ in enumerate(satellites):
            for dst_gid in range(num_gs):
                dst_gs_node_id = dst_gid + len(satellites)
                next_hop_decision = (-1, -1, -1)
                if forward_cost_sat_to_gs[curr_sid][dst_gid] != math.inf:
                    next_hop = forward_table_sat_to_gs[curr_sid][dst_gid]
                    if next_hop != dst_gs_node_id and sat_net_graph_only_satellites_with_isls.has_edge(curr_sid,next_hop) and \
                                    sat_net_graph_only_satellites_with_isls[curr_sid][next_hop]["weight"]!=math.inf:
                        next_hop_decision=(
                            next_hop,
                            sat_neighbor_to_if[curr_sid,next_hop],
                            sat_neighbor_to_if[next_hop,curr_sid]
                        )
                    elif next_hop == dst_gs_node_id:
                        next_hop_decision = (
                            dst_gs_node_id,
                            num_isls_per_sat[curr_sid] + gid_to_sat_gsl_if_idx[dst_gid],
                            0
                        )

                # Write to forwarding state
                if not prev_fstate or prev_fstate[(curr_sid, dst_gs_node_id)] != next_hop_decision:
                    f_out.write("%d,%d,%d,%d,%d\n" % (
                        curr_sid,
                        dst_gs_node_id,
                        next_hop_decision[0],
                        next_hop_decision[1],
                        next_hop_decision[2]
                    ))
                fstate[(curr_sid, dst_gs_node_id)] = next_hop_decision

        for src_gid in range (num_gs):
            for dst_gid in range(num_gs):
                if src_gid != dst_gid:
                    src_gs_node_id = len(satellites) + src_gid
                    dst_gs_node_id = len(satellites) + dst_gid
                    # 默认为源地面站无可接入卫星，或目标地面站无可达卫星，即 next_hop_decision 均为 -1
                    next_hop_decision = (-1, -1, -1)

                    if forward_cost_gs_to_gs[src_gid][dst_gid] != math.inf:
                        next_hop_decision = (
                            forward_table_gs_to_gs[src_gid][dst_gid],
                            0,
                            num_isls_per_sat[forward_table_gs_to_gs[src_gid][dst_gid]] + gid_to_sat_gsl_if_idx[src_gid]
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

    return {"fstate": fstate}



def generate_dynamic_state_at(
        output_dynamic_state_dir,
        epoch,
        time_since_epoch_ns,
        num_orbs,
        num_sats_per_orbs,
        satellites,
        ground_stations,
        list_isls,
        list_gsl_interfaces_info,
        max_gsl_length_m,
        max_isl_length_m,
        dynamic_state_algorithm,
        prev_output,
        enable_verbose_logs
):
    
    if enable_verbose_logs:
        print("FORWARDING STATE AT T = " + (str(time_since_epoch_ns))
              + "ns (= " + str(time_since_epoch_ns / 1e9) + " seconds)")

    #################################

    if enable_verbose_logs:
        print("\nBASIC INFORMATION")

    # Time
    time = epoch + time_since_epoch_ns * u.ns
    if enable_verbose_logs:
        print("  > Epoch.................. " + str(epoch))
        print("  > Time since epoch....... " + str(time_since_epoch_ns) + " ns")
        print("  > Absolute time.......... " + str(time))

    # Graphs
    sat_net_graph_only_satellites_with_isls = nx.Graph()
    sat_net_graph_all_with_only_gsls = nx.Graph()

    # Information
    for i in range(len(satellites)):
        sat_net_graph_only_satellites_with_isls.add_node(i)
        sat_net_graph_all_with_only_gsls.add_node(i)
    for i in range(len(satellites) + len(ground_stations)):
        sat_net_graph_all_with_only_gsls.add_node(i)
    if enable_verbose_logs:
        print("  > Satellites............. " + str(len(satellites)))
        print("  > Ground stations........ " + str(len(ground_stations)))
        print("  > Max. range GSL......... " + str(max_gsl_length_m) + "m")
        print("  > Max. range ISL......... " + str(max_isl_length_m) + "m")

    #################################

    if enable_verbose_logs:
        print("\nISL INFORMATION")

    # ISL edges
    total_num_isls = 0
    num_isls_per_sat = [0] * len(satellites)
    sat_neighbor_to_if = {}
    for (a, b) in list_isls:

        # ISLs are not permitted to exceed their maximum distance
        # TODO: Technically, they can (could just be ignored by forwarding state calculation),
        # TODO: but practically, defining a permanent ISL between two satellites which
        # TODO: can go out of distance is generally unwanted
        sat_distance_m = distance_m_between_satellites(satellites[a], satellites[b], str(epoch), str(time))
        if sat_distance_m > max_isl_length_m:
            raise ValueError(
                "The distance between two satellites (%d and %d) "
                "with an ISL exceeded the maximum ISL length (%.2fm > %.2fm at t=%dns)"
                % (a, b, sat_distance_m, max_isl_length_m, time_since_epoch_ns)
            )

        # Add to networkx graph
        sat_net_graph_only_satellites_with_isls.add_edge(
            a, b, weight=sat_distance_m
        )

        # Interface mapping of ISLs
        sat_neighbor_to_if[(a, b)] = num_isls_per_sat[a]
        sat_neighbor_to_if[(b, a)] = num_isls_per_sat[b]
        num_isls_per_sat[a] += 1
        num_isls_per_sat[b] += 1
        total_num_isls += 1

    if enable_verbose_logs:
        print("  > Total ISLs............. " + str(len(list_isls)))
        print("  > Min. ISLs/satellite.... " + str(np.min(num_isls_per_sat)))
        print("  > Max. ISLs/satellite.... " + str(np.max(num_isls_per_sat)))

    #################################

    if enable_verbose_logs:
        print("\nGSL INTERFACE INFORMATION")

    satellite_gsl_if_count_list = list(map(
        lambda x: x["number_of_interfaces"],
        list_gsl_interfaces_info[0:len(satellites)]
    ))
    ground_station_gsl_if_count_list = list(map(
        lambda x: x["number_of_interfaces"],
        list_gsl_interfaces_info[len(satellites):(len(satellites) + len(ground_stations))]
    ))
    if enable_verbose_logs:
        print("  > Min. GSL IFs/satellite........ " + str(np.min(satellite_gsl_if_count_list)))
        print("  > Max. GSL IFs/satellite........ " + str(np.max(satellite_gsl_if_count_list)))
        print("  > Min. GSL IFs/ground station... " + str(np.min(ground_station_gsl_if_count_list)))
        print("  > Max. GSL IFs/ground_station... " + str(np.max(ground_station_gsl_if_count_list)))

    #################################

    if enable_verbose_logs:
        print("\nGSL IN-RANGE INFORMATION")

    # What satellites can a ground station see
    ground_station_satellites_in_range = []
    for ground_station in ground_stations:
        # Find satellites in range
        satellites_in_range = []
        for sid in range(len(satellites)):
            distance_m = distance_m_ground_station_to_satellite(
                ground_station,
                satellites[sid],
                str(epoch),
                str(time)
            )
            if distance_m <= max_gsl_length_m:
                satellites_in_range.append((distance_m, sid))
                sat_net_graph_all_with_only_gsls.add_edge(
                    sid, len(satellites) + ground_station["gid"], weight=distance_m
                )

        ground_station_satellites_in_range.append(satellites_in_range)

    # Print how many are in range
    ground_station_num_in_range = list(map(lambda x: len(x), ground_station_satellites_in_range))
    if enable_verbose_logs:
        print("  > Min. satellites in range... " + str(np.min(ground_station_num_in_range)))
        print("  > Max. satellites in range... " + str(np.max(ground_station_num_in_range)))

    #################################

    #
    # Call the dynamic state algorithm which:
    #
    # (a) Output the gsl_if_bandwidth_<t>.txt files
    # (b) Output the fstate_<t>.txt files
    #
    if dynamic_state_algorithm == "algorithm_free_one_only_over_isls":

        return algorithm_free_one_only_over_isls(
            output_dynamic_state_dir,
            time_since_epoch_ns,
            num_orbs,
            num_sats_per_orbs,
            satellites,
            ground_stations,
            sat_net_graph_only_satellites_with_isls,
            ground_station_satellites_in_range,
            num_isls_per_sat,
            sat_neighbor_to_if,
            list_gsl_interfaces_info,
            prev_output,
            enable_verbose_logs
        )

    elif dynamic_state_algorithm == "algorithm_free_gs_one_sat_many_only_over_isls":

        return algorithm_free_gs_one_sat_many_only_over_isls(
            output_dynamic_state_dir,
            time_since_epoch_ns,
            satellites,
            ground_stations,
            sat_net_graph_only_satellites_with_isls,
            ground_station_satellites_in_range,
            num_isls_per_sat,
            sat_neighbor_to_if,
            list_gsl_interfaces_info,
            prev_output,
            enable_verbose_logs
        )

    elif dynamic_state_algorithm == "algorithm_free_one_only_gs_relays":

        return algorithm_free_one_only_gs_relays(
            output_dynamic_state_dir,
            time_since_epoch_ns,
            satellites,
            ground_stations,
            sat_net_graph_all_with_only_gsls,
            num_isls_per_sat,
            list_gsl_interfaces_info,
            prev_output,
            enable_verbose_logs
        )

    elif dynamic_state_algorithm == "algorithm_paired_many_only_over_isls":

        return algorithm_paired_many_only_over_isls(
            output_dynamic_state_dir,
            time_since_epoch_ns,
            satellites,
            ground_stations,
            sat_net_graph_only_satellites_with_isls,
            ground_station_satellites_in_range,
            num_isls_per_sat,
            sat_neighbor_to_if,
            list_gsl_interfaces_info,
            prev_output,
            enable_verbose_logs
        )

    else:
        raise ValueError("Unknown dynamic state algorithm: " + str(dynamic_state_algorithm))
