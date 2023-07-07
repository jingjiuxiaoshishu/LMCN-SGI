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

from satgen.isls import *
from satgen.ground_stations import *
from satgen.tles import *
from satgen.interfaces import *
from .generate_dynamic_state import generate_dynamic_state
import os
import math
from multiprocessing import Pool

def worker(args) :
    (
        output_generated_data_dir, 
        name, 
        simulation_end_time_ns,
        time_step_ns,
        offset_ns,
        duration_s,
        max_gsl_length_m, 
        max_isl_length_m, 
        dynamic_state_algorithm, 
        print_logs
     ) = args
    # 生成 generate_dynamic_state 所需参数
    args_list = generate_single_arg_list(
        output_generated_data_dir,
        name,
        simulation_end_time_ns,
        time_step_ns,
        offset_ns,
        duration_s,
        max_gsl_length_m,
        max_isl_length_m,
        dynamic_state_algorithm,
        print_logs
    )

    (
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
        dynamic_state_algorithm,
        print_logs
     ) = args_list
    
    
    # Generate dynamic state
    generate_dynamic_state(
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
                                  # "algorithm_free_gs_one_sat_many_only_over_isls"
                                  # "algorithm_paired_many_only_over_isls"
        print_logs
    )


def generate_single_arg_list(
        output_generated_data_dir,
        name,
        simulation_end_time_ns,
        time_step_ns,
        offset_ns,
        duration_s,
        max_gsl_length_m,
        max_isl_length_m,
        dynamic_state_algorithm,
        print_logs
):
    # Directory
    output_dynamic_state_dir = output_generated_data_dir + "/" + name + "/dynamic_state_" + str(int(time_step_ns/1000/1000)) \
                               + "ms_for_" + str(duration_s) + "s"
    # 若没有该输出文件夹，则新建文件夹
    # if not os.path.isdir(output_dynamic_state_dir):
    os.makedirs(output_dynamic_state_dir,exist_ok=True)


    # Variables (load in for each thread such that they don't interfere)
    ground_stations = read_ground_stations_extended(output_generated_data_dir + "/" + name + "/ground_stations.txt")
    tles = read_tles(output_generated_data_dir + "/" + name + "/tles.txt")
    satellites = tles["satellites"]
    num_orbs = int(tles["n_orbits"])
    num_sats_per_orbs = int(tles["n_sats_per_orbit"])
    list_isls = read_isls(output_generated_data_dir + "/" + name + "/isls.txt", len(satellites))
    list_gsl_interfaces_info = read_gsl_interfaces_info(
        output_generated_data_dir + "/" + name + "/gsl_interfaces_info.txt",
        len(satellites),
        len(ground_stations)
    )
    epoch = tles["epoch"]

    return (
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
        dynamic_state_algorithm,
        print_logs
    )


def generate_all_args_list(
        output_generated_data_dir, num_threads, name, time_step_ms, duration_s,
        max_gsl_length_m, max_isl_length_m, dynamic_state_algorithm, print_logs
):


    # In nanoseconds
    simulation_end_time_ns = duration_s * 1000 * 1000 * 1000
    time_step_ns = time_step_ms * 1000 * 1000

    # 单个任务默认生成连续 50 个时隙的路由表
    num_calculations_per_task = 50
    # 总共有多少个时隙的路由表需要计算
    num_calculations = math.floor(simulation_end_time_ns / time_step_ns)
    # 总共有多少个任务，即 list_args 的长度（向下取整）
    num_task = int(math.floor(float(num_calculations) / float(num_calculations_per_task)))
    # 最后一个任务需要计算的时隙路由表的个数
    num_calculations_last_task = num_calculations_per_task + num_calculations % num_calculations_per_task

    # Prepare arguments
    current = 0
    list_args = []
    for i in range(num_task):

        # How many time steps to calculate for
        if i < num_task-1:
            num_time_steps = num_calculations_per_task
        else:
            num_time_steps = num_calculations_last_task

        list_args.append((
            output_generated_data_dir,
            name,
            # 推测 (time_step_ns if (i + 1) != num_task else 0) 作用为保证实际计算时间成语仿真时长。
            (current + num_time_steps) * time_step_ns + (time_step_ns if (i + 1) != num_task else 0),
            time_step_ns,
            current * time_step_ns,
            duration_s,
            max_gsl_length_m,
            max_isl_length_m,
            dynamic_state_algorithm,
            print_logs
        ))

        current += num_time_steps
    return list_args


def help_dynamic_state(
        output_generated_data_dir, num_threads, name, time_step_ms, duration_s,
        max_gsl_length_m, max_isl_length_m, dynamic_state_algorithm, print_logs
):
    # build all args
    all_args = generate_all_args_list(
        output_generated_data_dir,
        num_threads,
        name,
        time_step_ms,
        duration_s,
        max_gsl_length_m,
        max_isl_length_m,
        dynamic_state_algorithm,
        print_logs
    )
    # Run in parallel
    with Pool(num_threads) as p:
        result = p.imap_unordered(worker,all_args,chunksize=1)
        # result 是一个 iterable，需要用 for 循环遍历才会执行
        for _ in result:
            pass
    # pool = ThreadPool(num_threads)
    # pool.map(worker, list_args)
    # pool.close()
    # pool.join()
