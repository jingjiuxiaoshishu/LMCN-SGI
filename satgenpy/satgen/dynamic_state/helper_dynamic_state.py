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


def help_dynamic_state(
        output_generated_data_dir, num_threads, name, time_step_ms, duration_s,
        max_gsl_length_m, max_isl_length_m, dynamic_state_algorithm, print_logs
):
    # In nanoseconds
    simulation_end_time_ns = duration_s * 1000 * 1000 * 1000
    time_step_ns = time_step_ms * 1000 * 1000
    offset_ns = 0


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
