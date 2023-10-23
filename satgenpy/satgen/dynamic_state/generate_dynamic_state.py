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

# liu:导入distribute中的包
from .distrubute_route.satellite_node import Satellite_node
from .distrubute_route.visible_helper import Visible_time_helper
from .distrubute_route.sat_selector import Sat_selector


def generate_dynamic_state(
        output_dynamic_state_dir,
        epoch,
        simulation_end_time_ns,
        time_step_ns,
        offset_ns,
        num_orbs,   # 新增参数
        num_sats_per_orbs,  # 新增参数
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
    # liu:
    # plus_grid_graph:卫星根据收集信息自己记录的拓扑状态
    # sat_net_graph_only_satellites_with_isls:记录实际星上拓扑，
    # 作为卫星之间通信是否可达的依据，节点之间权值为实际的distance，计算发送msg需要的时间
    sat_net_graph_only_satellites_with_isls = nx.Graph()
    plus_grid_graph = nx.Graph()

    for sid_1,sid_2 in list_isls:   # liu:对每一对ISL，计算两颗卫星之间的距离
        sat_distance_m = distance_m_between_satellites(satellites[sid_1], satellites[sid_2], str(epoch), str(epoch))
        sat_net_graph_only_satellites_with_isls.add_edge(sid_1,sid_2,weight = sat_distance_m)   # liu:有距离的星上拓扑
        plus_grid_graph.add_edge(sid_1,sid_2,weight = 1)    # liu:边为1的星上拓扑
        # liu:对于add_edge，如果涉及的节点不存在，则会自动添加节点，因此前面没有单独add_node()

    # liu:用于失效边 实验按照0.01计算，坏边的实际拓扑距离设置为math.inf
    import random
    # 百分之一的坏边
    print("\n 随机构建坏边，坏边率默认为百分之 1")
    percentage = 0
    num_fail_edges = int(percentage * len(list_isls))
    fail_edges = set(random.sample(list_isls, num_fail_edges))
    for fail_edge in fail_edges:
        sat_net_graph_only_satellites_with_isls[fail_edge[0]][fail_edge[1]]["weight"] = math.inf

    
    time_step_ms = time_step_ns/1000/1000   # liu: time_step转换为ms
    simulation_end_time_s = simulation_end_time_ns/1000/1000/1000   # liu: 总仿真时间转换为s


    print("\n 注意 astropy 的 Time 和 ephem.Date 的转换")   #liu:astropy和ephem都是python的天文计算库
    ephem_epoch = ephem.Date(epoch.datetime)    
    # liu:将给定的 epoch.datetime（一个 Python datetime 对象）
    # 转换为 ephem 库中的日期对象，并将结果存储在 ephem_epoch 变量中，以便后续进行天文计算或其他操作。

    # 建立卫星的虚拟节点
    print("\n satellite_nodes 建立中")   
    satellite_nodes = []
    # liu:enumerate用于同时获取列表中的元素和它们的索引，sid是索引，_是占位符，表示不使用元素的值
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
        # liu:这个文件是上面注释掉的部分生成的，见class:visible_time_helper.visible_times
        # liu:visible_times是卫星相对于地面站的可见时间，是一个二维字典，第一维为 gid，第二维为 sid，值为可见时间段[start, end]

    print("\n 建立选星器，并初始化 gsl ")
    shift_between_last_and_first = 8    # liu:not used
    
    # liu:创建选星器，初始化gsl
    sat_selector = Sat_selector(visible_times,num_orbs,num_sats_per_orbs,shift_between_last_and_first,ephem_epoch,time_step_ms, simulation_end_time_s)

    print("\n 添加更新事件，表明 epoch 需要对每个 gs 的 gsl 更新两次 ")
    for gid,gsl in sat_selector.gsls.items():
        # 添加更新事件，表明 epoch 需要对每个 gs 的 gsl 更新两次
        for i in range(2):
            sat_selector.add_gsl_update_event(gid,i,0)
            # liu:
            # def add_gsl_update_event(self,gid,gsl_to_update,slot_to_update_gsl)
            # gsl_to_update为i，slot_to_update_gsl为0，表示在第0个时隙更新gsl
            # 在第0个时隙在time_event_scheduler中为两条gsl添加gsl更新事件

    print(f"\n 初始化 gsl，此时 gsl 均为 -1 \n 初始化 forward_table_to_gsf" "\n 初始化 forward_table_to_sats"  "\n 初始化 init_state_of_edges")
    for satellite_node in satellite_nodes:
        satellite_node.init_forward_table_to_gs(len(sat_selector.gsls)) # liu:初始化gs之间的路由表，每两个gs之间的forward_table和cost为-1
        satellite_node.init_forward_table_to_sats(len(satellite_nodes)) # liu:初始化卫星到卫星的路由表
        satellite_node.init_state_of_edges()    # liu:初始化边状态为true（好边）
        satellite_node.update_forward_table_to_sats()   # liu:根据卫星的self.satellite_topo计算从self.sid到其他卫星的最短路径


    prev_output = None
    i = 0
    total_iterations = ((simulation_end_time_ns - offset_ns) / time_step_ns)    # liu:总迭代次数
    # liu:每个时隙，执行如下循环
    for time_since_epoch_ns in range(offset_ns, simulation_end_time_ns, time_step_ns):
        if not enable_verbose_logs:
            if i % int(math.floor(total_iterations) / 10.0) == 0:
                print("Progress: calculating for T=%d (time step granularity is still %d ms)" % (
                    time_since_epoch_ns, time_step_ns / 1000000
                ))
            i += 1

        
        # sat_net_graph_only_satellites_with_isls 更新（实际拓扑，两个节点之间的权值设置为实际拓扑，坏边为math.inf）
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
        sat_selector.process()  # liu:处理上述添加的gsl更新msg，更新gsl（1. 更新gsl）


        print("\n satellite_node.process() \n 依据gsl是否变化和卫星自身路由表是否变化，确定是否需要更新 gsl 及对应路由表")
        import threading
        def worker(satellite_nodes,start,stop,sat_net_graph_only_satellites_with_isls,sat_selector):
            for satellite_node in satellite_nodes[start:stop]:
                satellite_node.update_sat_net_graph_only_satellites_with_isls(sat_net_graph_only_satellites_with_isls)  # liu:deepcopy
                satellite_node.process()    # liu:处理msg，如果need_to_update_forward_table_to_sats，则更新卫星到其他卫星的路由表（2. 更新星上路由表）
                # liu: 关于process：处理每个时隙卫星必要做的一些事情，包括：
                # 1.遍历卫星的消息队列，处理消息
                # 2.处理边状态更新
                # 3.定时发送hello消息确认链路状态
                # 4.定时发送fail消息，通知链路失效

                # 不管有没有变动，每个时隙都复制一遍 sat_selector 的 gsls，并重新计算 sat to gs
                if  satellite_node.need_to_update_forward_table_to_sats:
                    for gid,gsl in sat_selector.gsls.items():
                        for i in range(2):
                            satellite_node.update_gsl(gid,gsl[i],i) 
                            # liu:这里更新地面站gid的第i条gsl连接的卫星为gsl[i]，然后更新从卫星satellite_node到gid的路由表（3.更新卫星到地面站）
                elif len(sat_selector.time_event_scheduler[sat_selector.curr_slot - 1]) > 0:
                    for msg in sat_selector.time_event_scheduler[sat_selector.curr_slot - 1]:
                        gid = msg["gid"]
                        gsl_to_update = msg["gsl_to_update"]
                        satellite_node.update_gsl(gid,sat_selector.gsls[gid][gsl_to_update],gsl_to_update)
                        # liu:更新地面站gid的gsl_to_update连接的卫星为sat_selector.gsls[gid][gsl_to_update]，然后更新卫星到该地面站的路由表

        print("\n 构建多线程参数")
        threads = []    # liu:存放线程对象
        satellite_node_slices = []  # liu:存放分割的satellite_nodes的索引范围
        slices_step = 330   # liu:步长，决定每一线程处理多少satellite_nodes中的元素
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

        # liu:每一个分片的索引范围，都创建一个新的线程。这个线程的目标函数是worker
        for start,stop in satellite_node_slices:
            thread = threading.Thread(target=worker, kwargs={"satellite_nodes": satellite_nodes, 
                                                             "start":start,
                                                             "stop":stop,
                                                             "sat_net_graph_only_satellites_with_isls": sat_net_graph_only_satellites_with_isls,
                                                             "sat_selector":sat_selector})
            threads.append(thread)

        # liu:启动所有线程，并发执行
        print("调度线程")
        for thread in threads:
            thread.start()

        # liu:join，等待所有子线程的完成
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
        print("\nISL INFORMATION")  # liu:从ISL INFORMATION开始，之前的部分如设置图，添加节点，在前面代码都设置好了

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

    # liu:获取list_gsl_interfaces_info的前len个数据（从gsl_interfaces_info.txt读取的）
    # 使用map函数和一个lambda匿名函数，从每个字典中提取number_of_interfaces的值。
    # 将结果转换为一个列表并存储到satellite_gsl_if_count_list中。
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


    # liu:这里直接把algorithm_free_one_only_over_isls中的内容写到这里，原本需要根据dynamic_state_algorithm进行算法选择
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
                forward_table_gs_to_gs[gid_1][gid_2] = -1   # liu:默认不可达
                forward_cost_gs_to_gs[gid_1][gid_2] = math.inf  # liu:默认成本为inf

    # liu:根据 gsls 和 卫星原本拥有的 sat to gs 路由表，计算 gs to gs 的路由表(4.更新地面站到地面站的路由表)
    for gid_1 in range(len(gsls)):
        for gid_2 in range(len(gsls)):
            if gid_1 != gid_2:
                next_hop = -1
                cost = math.inf
                for sid in sat_selector.gsls[gid_1]:    # liu:找到与gid_1相连的卫星，根据sat_to_gs路由表计算地面站之间的路由
                    if sid != -1:
                        if satellite_nodes[sid].forward_cost_to_gs[gid_2] < cost:
                            next_hop = sid
                            cost = satellite_nodes[sid].forward_cost_to_gs[gid_2]
                forward_table_gs_to_gs[gid_1][gid_2] = next_hop
                forward_cost_gs_to_gs[gid_1][gid_2] = cost + 1

    # Forwarding state  liu:生成forwarding state路由表
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


# liu:这个是hypatia原本的函数，没有改动
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
