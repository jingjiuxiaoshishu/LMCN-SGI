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
    percentage = 0
    num_fail_edges = int(percentage * len(list_isls))
    fail_edges = set(random.sample(list_isls, num_fail_edges))
    for fail_edge in fail_edges:
        sat_net_graph_only_satellites_with_isls[fail_edge[0]][fail_edge[1]]["weight"] = math.inf

    
    time_step_ms = time_step_ns/1000/1000
    simulation_end_time_s = simulation_end_time_ns/1000/1000/1000


    print("\n 注意 astropy 的 Time 和 ephem.Date 的转换")
    ephem_epoch = ephem.Date(epoch.datetime)

    # 建立卫星的虚拟节点
    print("\n satellite_nodes 建立中")   
    satellite_nodes = []
    for sid,_ in enumerate(satellites):
        satellite_node = Satellite_node(sid, satellites[sid], plus_grid_graph,
                                    sat_net_graph_only_satellites_with_isls,ephem_epoch,time_step_ms, simulation_end_time_s)
        satellite_nodes.append(satellite_node)

    # # 建立地面站观察者
    # print("\n ground_observers 建立中")    
    # ground_observers = []
    # for ground_station in ground_stations:
    #     ground_observer = ephem.Observer()
    #     ground_observer.lat = ground_station["latitude_degrees_str"]
    #     ground_observer.lon = ground_station["longitude_degrees_str"]
    #     ground_observers.append(ground_observer)
        
    # print("\n 可见时间计算中,请注意最长仿真时间不要超过卫星的周期")
    # visible_time_helper = Visible_time_helper(ground_observers, satellites, 25, ephem_epoch,
    #                                           time_step_ms, simulation_end_time_s)
    
    # import pickle
    # visible_times = visible_time_helper.visible_times
    # with open("visible_times.pkl","wb") as f:
    #     pickle.dump(visible_times,f)

    import pickle
    with open("visible_times.pkl","rb") as f:
        visible_times = pickle.load(f)

    print("\n 建立选星器，并初始化 gsl ")
    shift_between_last_and_first = 8
    
    sat_selector = Sat_selector(visible_times,num_orbs,num_sats_per_orbs,shift_between_last_and_first,ephem_epoch,time_step_ms, simulation_end_time_s)

    print("\n 添加更新事件，表明 epoch 需要对每个 gs 的 gsl 更新两次 ")
    for gid,gsl in sat_selector.gsls.items():
        # 添加更新事件，表明 epoch 需要对每个 gs 的 gsl 更新两次
        for i in range(2):
            sat_selector.add_gsl_update_event(gid,i,0)

    print(f"\n 初始化 gsl，此时 gsl 均为 -1 \n 初始化 forward_table_to_gs" "\n 初始化 forward_table_to_sats"  "\n 初始化 init_state_of_edges")
    for satellite_node in satellite_nodes:
        satellite_node.init_forward_table_to_gs(len(sat_selector.gsls))
        satellite_node.init_forward_table_to_sats(len(satellite_nodes))
        satellite_node.init_state_of_edges()
        satellite_node.update_forward_table_to_sats()


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
        for src,dst,attrs in sat_net_graph_only_satellites_with_isls.edges(data=True):
            edge = (src,dst)
            edge_weight = attrs["weight"]
            if edge_weight != math.inf:
                distance = distance_m_between_satellites(satellite_nodes[edge[0]].satellite, satellite_nodes[edge[1]].satellite, str(epoch),  str(epoch + time_since_epoch_ns*u.ns))
                sat_net_graph_only_satellites_with_isls[edge[0]][edge[1]]["weight"] = distance

        
        # 注意顺序
        # 1、更新 gsl
        # 2、更新星上路由表 ( 同时会把 sat_net_graph_only_satellites_with_isls 更新同步)
        # 3、更新卫星到地面站
        # 4、更新地面站到地面站 （ 在 generate_dynamic_state_distribute 里更新）
        # 5、写入路由表等
        # 6、sat_net_graph_only_satellites_with_isls 更新

        print("\n sat_selector.process()")
        sat_selector.process()


        print("\n satellite_node.process() \n 依据gsl是否变化和卫星自身路由表是否变化，确定是否需要更新 gsl 及对应路由表")
        import threading
        def worker(satellite_nodes,start,stop,sat_net_graph_only_satellites_with_isls,sat_selector):
            for satellite_node in satellite_nodes[start:stop]:
                satellite_node.update_sat_net_graph_only_satellites_with_isls(sat_net_graph_only_satellites_with_isls)
                satellite_node.process()
                # 不管有没有变动，每个时隙都复制一遍 sat_selector 的 gsls，并重新计算 sat to gs
                if  satellite_node.need_to_update_forward_table_to_sats:
                    for gid,gsl in sat_selector.gsls.items():
                        for i in range(2):
                            satellite_node.update_gsl(gid,gsl[i],i)
                elif len(sat_selector.time_event_scheduler[sat_selector.curr_slot - 1]) > 0:
                    for msg in sat_selector.time_event_scheduler[sat_selector.curr_slot - 1]:
                        gid = msg["gid"]
                        gsl_to_update = msg["gsl_to_update"]
                        satellite_node.update_gsl(gid,sat_selector.gsls[gid][gsl_to_update],gsl_to_update)

        print("\n 构建多线程参数")
        threads = []
        satellite_node_slices = []
        slices_step = 330
        start = 0
        stop = start + slices_step
        while stop < len(satellite_nodes):
            satellite_node_slices.append((start,stop))
            start = start + slices_step
            stop = stop + slices_step
            if stop > len(satellite_nodes):
                stop = len(satellite_nodes)
        if start!=len(satellite_nodes):
            satellite_node_slices.append((start,stop))
        
        print("建立线程")
        for start,stop in satellite_node_slices:
            thread = threading.Thread(target=worker, kwargs={"satellite_nodes": satellite_nodes, 
                                                             "start":start,
                                                             "stop":stop,
                                                             "sat_net_graph_only_satellites_with_isls": sat_net_graph_only_satellites_with_isls,
                                                             "sat_selector":sat_selector})
            threads.append(thread)
        print("调度线程")
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        print("全部线程执行完毕")




        # print("\n satellite_node.process() \n 依据gsl是否变化和卫星自身路由表是否变化，确定是否需要更新 gsl 及对应路由表")
        # for satellite_node in satellite_nodes:
        #     satellite_node.update_sat_net_graph_only_satellites_with_isls(sat_net_graph_only_satellites_with_isls)
        #     satellite_node.process()
        #     # 不管有没有变动，每个时隙都复制一遍 sat_selector 的 gsls，并重新计算 sat to gs
        #     if  satellite_node.need_to_update_forward_table_to_sats:
        #         for gid,gsl in sat_selector.gsls.items():
        #             for i in range(2):
        #                 satellite_node.update_gsl(gid,gsl[i],i)
        #     elif len(sat_selector.time_event_scheduler[sat_selector.curr_slot - 1]) > 0:
        #         for msg in sat_selector.time_event_scheduler[sat_selector.curr_slot - 1]:
        #             gid = msg["gid"]
        #             gsl_to_update = msg["gsl_to_update"]
        #             satellite_node.update_gsl(gid,sat_selector.gsls[gid][gsl_to_update],gsl_to_update)

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
    
    prev_fstate = None
    if prev_output is not None:
        prev_fstate = prev_output["fstate"]
    
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



    print("写入 gsl 接口的带宽 gsl_if_bandwidth_")
    output_filename = output_dynamic_state_dir + "/gsl_if_bandwidth_" + str(time_since_epoch_ns) + ".txt"
    if enable_verbose_logs:
        print("  > Writing interface bandwidth state to: " + output_filename)
    with open(output_filename, "w+") as f_out:
        if time_since_epoch_ns == 0:
            for node_id in range(len(satellite_nodes)):
                f_out.write("%d,%d,%f\n"
                            % (node_id, num_isls_per_sat[node_id],
                               list_gsl_interfaces_info[node_id]["aggregate_max_bandwidth"]))
            for node_id in range(len(satellite_nodes), len(satellite_nodes) + len(sat_selector.gsls)):
                f_out.write("%d,%d,%f\n"
                            % (node_id, 0, list_gsl_interfaces_info[node_id]["aggregate_max_bandwidth"]))


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
                if prev_fstate:
                    next_hop_1 =  prev_fstate[(gid_1+len(satellite_nodes),gid_2+len(satellite_nodes))][0]
                    next_hop_2 =  prev_fstate[(gid_2+len(satellite_nodes),gid_1+len(satellite_nodes))][0]
                    if next_hop_1 == -1 or next_hop_2 == -1:
                        link_exist = -1
                    else:
                        link_exist = satellite_nodes[next_hop_1].forward_table_to_sats[next_hop_2] 
                else:
                    next_hop_1 =  -1
                    next_hop_2 =  -1
                    link_exist = -1
                # 如果两个 pre_gsl 均可用，且两颗接入卫星之间有路径。
                if (next_hop_1 in satellite_nodes[next_hop_1].gsl[gid_1]) and next_hop_1!=-1 and (next_hop_2 in satellite_nodes[next_hop_2].gsl[gid_2]) and next_hop_2!=-1 and link_exist!=-1:
                    forward_table_gs_to_gs[gid_1][gid_2] = next_hop_1
                    forward_cost_gs_to_gs[gid_1][gid_2] = satellite_nodes[next_hop_1].forward_cost_to_gs[gid_2] + 1
                else:
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
    gid_to_sat_gsl_if_idx = [0] * len(gsls) 

    # Now write state to file for complete graph
    output_filename = output_dynamic_state_dir + "/fstate_" + str(time_since_epoch_ns) + ".txt"
    if enable_verbose_logs:
        print("  > Writing forwarding state to: " + output_filename)
    with open(output_filename, "w+") as f_out:
        for curr_sid,_ in enumerate(satellite_nodes):
            for dst_gid,_ in enumerate(gsls):
                dst_gs_node_id = dst_gid + len(satellite_nodes)
                next_hop_decision = (-1, -1, -1)
                if satellite_nodes[curr_sid].forward_cost_to_gs[dst_gid] != math.inf:
                    if satellite_nodes[curr_sid].forward_table_to_gs[dst_gid] != dst_gid:
                        next_hop_decision=(
                            satellite_nodes[curr_sid].forward_table_to_gs[dst_gid],
                            sat_neighbor_to_if[curr_sid,satellite_nodes[curr_sid].forward_table_to_gs[dst_gid]],
                            sat_neighbor_to_if[satellite_nodes[curr_sid].forward_table_to_gs[dst_gid],curr_sid]
                        )
                    else:
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

        for src_gid,_ in enumerate (gsls):
            for dst_gid,_ in enumerate(gsls):
                if src_gid != dst_gid:
                    src_gs_node_id = len(satellite_nodes) + src_gid
                    dst_gs_node_id = len(satellite_nodes) + dst_gid
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
