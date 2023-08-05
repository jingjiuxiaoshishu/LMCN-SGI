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
from .distrubute_route.sat_selector import Sat_selector


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
    plus_grid_graph = nx.Graph()

    for sid_1,sid_2 in list_isls:
        sat_distance_m = distance_m_between_satellites(satellites[sid_1], satellites[sid_2], str(epoch), str(epoch))
        sat_net_graph_only_satellites_with_isls.add_edge(sid_1,sid_2,weight = sat_distance_m)
        plus_grid_graph.add_edge(sid_1,sid_2,weight = 1)

    import random
    # 百分之一的坏边
    print("\n 随机构建坏边，坏边率默认为百分之 1")
    percentage = 0.01
    num_fail_edges = int(percentage * len(list_isls))
    fail_edges = set(random.sample(list_isls, num_fail_edges))
    for fail_edge in fail_edges:
        sat_net_graph_only_satellites_with_isls[fail_edge[0]][fail_edge[1]]["weight"] = math.inf

    
    time_step_ms = time_step_ns/1000/1000
    simulation_end_time_s = simulation_end_time_ns/1000/1000/1000

    # 建立卫星的虚拟节点
    print("\n satellite_nodes 建立中")   
    satellite_nodes = []
    for sid in range(len(satellites)):
        satellite_node = Satellite_node(sid, satellites[sid], plus_grid_graph,
                                    sat_net_graph_only_satellites_with_isls,epoch,time_step_ms, simulation_end_time_s)
        satellite_nodes.append(satellite_node)

    # 建立地面站观察者
    print("\n ground_observers 建立中")    
    ground_observers = []
    for ground_station in ground_stations:
        ground_observer = ephem.Observer()
        ground_observer.lat = ground_station["latitude_degrees_str"]
        ground_observer.lon = ground_station["longitude_degrees_str"]
        ground_observers.append(ground_observer)
        
    visible_time_helper = Visible_time_helper(ground_observers, satellites, 25, epoch,
                                              time_step_ms, simulation_end_time_s)
    
    visible_times = visible_time_helper.visible_times


    shift_between_last_and_first = 8
    sat_selector = Sat_selector(visible_times,num_orbs,num_sats_per_orbs,shift_between_last_and_first,epoch,time_step_ms, simulation_end_time_s)

    for satellite_node in satellite_nodes:
        satellite_node.init_forward_table_to_gs(len(sat_selector.gsls))
        for gid,gsl in sat_selector.gsls.items():
            for i in range(2):
                satellite_node.update_gsl(gid,gsl[i],i)



    
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

        

        sat_selector.process()
        for satellite_node in satellite_nodes:
            satellite_node.process()
            # 不管有没有变动，每个时隙都复制一遍 sat_selector 的 gsls，并重新计算 sat to gs
            for gid,gsl in sat_selector.gsls.items():
                for i in range(2):
                    satellite_node.update_gsl(gid,gsl[i],i)

        prev_output = generate_dynamic_state_distribute(
            epoch,
            satellite_nodes,
            sat_selector,
            output_dynamic_state_dir,
            time_since_epoch_ns,
            list_isls,
            max_isl_length_m,
            sat_net_graph_only_satellites_with_isls,
            list_gsl_interfaces_info,
            prev_output,
            enable_verbose_logs
        )

        for u,v,attrs in sat_net_graph_only_satellites_with_isls.edges(data=True):
            edge = (u,v)
            edge_weight = attrs["weight"]
            if edge_weight != math.inf:
                distance = distance_m_between_satellites(satellite_nodes[edge[0]].satellite, satellite_nodes[edge[1]].satellite, str(epoch), "2000/01/01 00:00:01")
                sat_net_graph_only_satellites_with_isls[edge[0]][edge[1]]["weight"] = distance

                



            
    

def generate_dynamic_state_distribute(
        epoch,
        satellite_nodes,
        sat_selector,
        output_dynamic_state_dir,
        time_since_epoch_ns,
        list_isls,
        max_isl_length_m,
        sat_net_graph_only_satellites_with_isls,
        list_gsl_interfaces_info,
        prev_output,
        enable_verbose_logs
):
    
    #################################
    if enable_verbose_logs:
        print("\nISL INFORMATION")

    # ISL edges
    total_num_isls = 0
    num_isls_per_sat = [0] * len(satellite_nodes)
    sat_neighbor_to_if = {}
    for (a, b) in list_isls:

        # ISLs are not permitted to exceed their maximum distance
        # TODO: Technically, they can (could just be ignored by forwarding state calculation),
        # TODO: but practically, defining a permanent ISL between two satellites which
        # TODO: can go out of distance is generally unwanted
        sat_distance_m = distance_m_between_satellites(satellite_nodes[a].satellite, satellite_nodes[b].satellite, str(epoch), str(epoch + time_since_epoch_ns*u.ns))
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
        list_gsl_interfaces_info[0:len(satellite_nodes)]
    ))
    ground_station_gsl_if_count_list = list(map(
        lambda x: x["number_of_interfaces"],
        list_gsl_interfaces_info[len(satellite_nodes):(len(satellite_nodes) + len(sat_selector.gsls))]
    ))
    if enable_verbose_logs:
        print("  > Min. GSL IFs/satellite........ " + str(np.min(satellite_gsl_if_count_list)))
        print("  > Max. GSL IFs/satellite........ " + str(np.max(satellite_gsl_if_count_list)))
        print("  > Min. GSL IFs/ground station... " + str(np.min(ground_station_gsl_if_count_list)))
        print("  > Max. GSL IFs/ground_station... " + str(np.max(ground_station_gsl_if_count_list)))


    #################################
    if enable_verbose_logs:
        print(f"\n 更新 {str(epoch + time_since_epoch_ns*u.ns)} 时刻路由表")
    
    gsls = sat_selector.gsls 
    # 初始化一个空的 gs 到 gs 的路由表
    forward_table_gs_to_gs = {}
    forward_cost_gs_to_gs = {}
    for gid_1 in range(len(gsls)):
        forward_table_gs_to_gs[gid_1] = {}
        forward_cost_gs_to_gs[gid_1] = {}
        for gid_2 in range(len(gsls)):
            if gid_1 != gid_2:
                forward_table_gs_to_gs[gid_1][gid_2] = -1
                forward_cost_gs_to_gs[gid_1][gid_2] = math.inf

    # 根据 gsls 和 卫星原本拥有的 sat to gs 路由表，计算 gs to gs 的路由表
    for gid_1 in range(len(gsls)):
        for gid_2 in range(len(gsls)):
            if gid_1 != gid_2:
                next_hop = -1
                cost = math.inf
                for sid in sat_selector.gsls[gid_1]:
                    if sid != -1:
                        if satellite_nodes[sid].forward_cost_to_gs[gid_2] < cost:
                            next_hop = sid
                            cost = satellite_nodes[sid].forward_cost_to_gs[gid_2]
                forward_table_gs_to_gs[gid_1][gid_2] = next_hop
                forward_cost_gs_to_gs[gid_1][gid_2] = cost + 1

    # Forwarding state
    fstate = {}

    # Now write state to file for complete graph
    output_filename = output_dynamic_state_dir + "/fstate_" + str(time_since_epoch_ns) + ".txt"
    if enable_verbose_logs:
        print("  > Writing forwarding state to: " + output_filename)
    with open(output_filename, "w+") as f_out:
        
        for curr_sid in range(len(satellite_nodes)):
            for dst_gid in range(len(gsls)):
                next_hop_decision = (-1, -1, -1)
                if satellite_nodes[curr_sid].forward_cost_to_gs[dst_gid] != math.inf:
                    next_hop_decision={
                        satellite_nodes[curr_sid].forward_table_to_gs[dst_gid],
                        sat_neighbor_to_if[curr_sid,satellite_nodes[curr_sid].forward_table_to_gs[dst_gid]],
                        sat_neighbor_to_if[satellite_nodes[curr_sid].forward_table_to_gs[dst_gid],curr_sid]
                    }

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
    

    return 1

    



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
