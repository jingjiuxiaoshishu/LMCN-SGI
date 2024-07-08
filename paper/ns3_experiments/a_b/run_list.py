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

# Core values
dynamic_state_update_interval_ms = 100                          # 100 millisecond update interval
simulation_end_time_s = 1000                                     # 200 seconds
pingmesh_interval_ns = 1 * 1000 * 1000                          # A ping every 1ms
enable_isl_utilization_tracking = True                          # Enable utilization tracking
isl_utilization_tracking_interval_ns = 1 * 1000 * 1000 * 1000   # 1 second utilization intervals

# Derivatives
dynamic_state_update_interval_ns = dynamic_state_update_interval_ms * 1000 * 1000
simulation_end_time_ns = simulation_end_time_s * 1000 * 1000 * 1000
dynamic_state = "dynamic_state_" + str(dynamic_state_update_interval_ms) + "ms_for_" + str(simulation_end_time_s) + "s"

# # Chosen pairs:
# # > Rio de Janeiro (1174) to St. Petersburg (1229)
# # > Manila (1173) to Dalian (1241)
# # > Istanbul (1170) to Nairobi (1252)
# # > Paris (1180 (1156 for the Paris-Moscow GS relays)) to Moscow (1177 (1232 for the Paris-Moscow GS relays))
# full_satellite_network_isls = "kuiper_630_isls_plus_grid_ground_stations_top_100_algorithm_free_one_only_over_isls"
# full_satellite_network_gs_relay = "kuiper_630_isls_none_ground_stations_paris_moscow_grid_algorithm_free_one_only_gs_relays"
# chosen_pairs = [
#     ("kuiper_630_isls", 1174, 1229, "TcpNewReno", full_satellite_network_isls),
#     ("kuiper_630_isls", 1174, 1229, "TcpVegas", full_satellite_network_isls),
#     ("kuiper_630_isls", 1173, 1241, "TcpNewReno", full_satellite_network_isls),
#     ("kuiper_630_isls", 1173, 1241, "TcpVegas", full_satellite_network_isls),
#     ("kuiper_630_isls", 1170, 1252, "TcpNewReno", full_satellite_network_isls),
#     ("kuiper_630_isls", 1170, 1252, "TcpVegas", full_satellite_network_isls),
#     ("kuiper_630_isls", 1180, 1177, "TcpNewReno", full_satellite_network_isls),
#     ("kuiper_630_gs_relays", 1156, 1232, "TcpNewReno", full_satellite_network_gs_relay),
# ]


# Chosen pairs:
# > Itaboraí(133) to Kaunas(116)            1296 + 133 = 1429     1296 + 116 = 1412
# > Apra Heights GU(120) to Ajigaura(162)   1296 + 120 = 1416     1296 + 162 = 1458
# > Villenave d’Ornon(72) to Lekki(152)     1296 + 72 = 1368      1296 + 152 = 1448

# full_satellite_network_isls = "starlink_550_isls_plus_grid_ground_stations_starlink_algorithm_free_one_only_over_isls"

# chosen_pairs = [
#     ("starlink_550_isls", 1429, 1412, "TcpNewReno", full_satellite_network_isls),
#     ("starlink_550_isls", 1429, 1412, "TcpVegas", full_satellite_network_isls),
#     ("starlink_550_isls", 1416, 1458, "TcpNewReno", full_satellite_network_isls),
#     ("starlink_550_isls", 1416, 1458, "TcpVegas", full_satellite_network_isls),
#     ("starlink_550_isls", 1368, 1448, "TcpNewReno", full_satellite_network_isls),
#     ("starlink_550_isls", 1368, 1448, "TcpVegas", full_satellite_network_isls),
# ]

# TCP Reno和Vegas应该是两种拥塞控制算法

full_satellite_network_isls = "new_starlink_550_isls_plus_grid_ground_stations_starlink_algorithm_free_one_only_over_isls"

# new experiment
chosen_pairs = [
    ("starlink_550_isls", 1428, 1433, "TcpHybla", full_satellite_network_isls), # Guarapari-Mossoró, Distance: 1735.6245179531911 km
    ("starlink_550_isls", 1380, 1429, "TcpHybla", full_satellite_network_isls), # Puerto Montt-Itaboraí, Distance: 3496.8870496604764 km
    ("starlink_550_isls", 1428, 1448, "TcpHybla", full_satellite_network_isls), # Guarapari-Lekki, Distance: 5663.13441959684 km
    ("starlink_550_isls", 1375, 1448, "TcpHybla", full_satellite_network_isls), # Noviciado-Lekki, Distance: 8981.135126517354 km
    ("starlink_550_isls", 1429, 1412, "TcpHybla", full_satellite_network_isls), # Itaboraí-Kaunas, Distance: 10663.65598168759 km
    ("starlink_550_isls", 1364, 1375, "TcpHybla", full_satellite_network_isls), # Willows QLD-Noviciado, Distance: 12500.392408137142 km
    ("starlink_550_isls", 1390, 1429, "TcpHybla", full_satellite_network_isls), # Springbrook Creek NSW-Itaboraí, Distance: 13974.337264882593 km
    ("starlink_550_isls", 1416, 1448, "TcpHybla", full_satellite_network_isls), # Apra Heights GU-Lekki, Distance: 15200.86463974216 km
    ("starlink_550_isls", 1416, 1443, "TcpHybla", full_satellite_network_isls), # Apra Heights GU-Falda del Carmen, Distance: 16434.845430798403 km
    ("starlink_550_isls", 1416, 1433, "TcpHybla", full_satellite_network_isls)  # Apra Heights GU-Mossoró, Distance: 19066.09839237275 km
]

# new2 experiment
chosen_pairs = [
    ("starlink_550_isls", 1444, 1454, "TcpHybla", full_satellite_network_isls), # Milano-Wherstead (LICENSE PENDING), Distance: 950.3259764044684 km
    ("starlink_550_isls", 1364, 1383, "TcpHybla", full_satellite_network_isls), # Willows QLD-Merredin WA, Distance: 3002.7283616981317 km
    ("starlink_550_isls", 1433, 1448, "TcpHybla", full_satellite_network_isls), # Mossoró-Lekki, Distance: 4727.013114577765 km
    ("starlink_550_isls", 1433, 1454, "TcpHybla", full_satellite_network_isls), # Mossoró-Wherstead (LICENSE PENDING), Distance: 7308.2753470908565 km
    ("starlink_550_isls", 1412, 1416, "TcpHybla", full_satellite_network_isls), # Kaunas-Apra Heights GU, Distance: 10640.403703179678 km
    ("starlink_550_isls", 1375, 1444, "TcpHybla", full_satellite_network_isls), # Noviciado-Milano, Distance: 11865.320976485178 km
    ("starlink_550_isls", 1383, 1448, "TcpHybla", full_satellite_network_isls), # Merredin WA-Lekki, Distance: 12725.376243957591 km
    ("starlink_550_isls", 1383, 1429, "TcpHybla", full_satellite_network_isls), # Merredin WA-Itaboraí, Distance: 13682.293455059284 km
    ("starlink_550_isls", 1364, 1429, "TcpHybla", full_satellite_network_isls), # Willows QLD-Itaboraí, Distance: 14754.597840542565 km
    ("starlink_550_isls", 1364, 1454, "TcpHybla", full_satellite_network_isls), # Willows QLD-Wherstead (LICENSE PENDING), Distance: 15762.313237292155 km
    ("starlink_550_isls", 1390, 1454, "TcpHybla", full_satellite_network_isls), # Springbrook Creek NSW-Wherstead (LICENSE PENDING), Distance: 16499.116954011977 km
    ("starlink_550_isls", 1364, 1433, "TcpHybla", full_satellite_network_isls), # Willows QLD-Mossoró, Distance: 16772.819724295878 km
    ("starlink_550_isls", 1416, 1429, "TcpHybla", full_satellite_network_isls), # Apra Heights GU-Itaboraí, Distance: 18713.378673993346 km
    ("starlink_550_isls", 1416, 1428, "TcpHybla", full_satellite_network_isls)  # Apra Heights GU-Guarapari, Distance: 19054.61390765489 km
]

# new3 experiment
chosen_pairs = [
    ("starlink_550_isls", 1367, 1383, "TcpHybla", full_satellite_network_isls), # Wagin WA-Merredin WA, Distance: 219.4654655365272 km
    ("starlink_550_isls", 1364, 1390, "TcpHybla", full_satellite_network_isls), # Willows QLD-Springbrook Creek NSW, Distance: 781.0095826293394 km
    ("starlink_550_isls", 1375, 1429, "TcpHybla", full_satellite_network_isls), # Noviciado-Itaboraí, Distance: 2985.74556717517 km
    ("starlink_550_isls", 1444, 1448, "TcpHybla", full_satellite_network_isls), # Milano-Lekki, Distance: 4341.6896948241065 km
    ("starlink_550_isls", 1429, 1448, "TcpHybla", full_satellite_network_isls), # Itaboraí-Lekki, Distance: 5997.9700529407455 km
    ("starlink_550_isls", 1380, 1448, "TcpHybla", full_satellite_network_isls), # Puerto Montt-Lekki, Distance: 9379.279610221876 km
    ("starlink_550_isls", 1375, 1454, "TcpHybla", full_satellite_network_isls), # Noviciado-Wherstead (LICENSE PENDING), Distance: 11762.33273324635 km
    ("starlink_550_isls", 1367, 1448, "TcpHybla", full_satellite_network_isls), # Wagin WA-Lekki, Distance: 12610.55360351455 km
    ("starlink_550_isls", 1367, 1429, "TcpHybla", full_satellite_network_isls), # Wagin WA-Itaboraí, Distance: 13463.059013719083 km
    ("starlink_550_isls", 1383, 1454, "TcpHybla", full_satellite_network_isls), # Merredin WA-Wherstead (LICENSE PENDING), Distance: 14515.808673092053 km
    ("starlink_550_isls", 1390, 1448, "TcpHybla", full_satellite_network_isls)  # Springbrook Creek NSW-Lekki, Distance: 15596.735173583394 km
]

# error paths：
# 1367 1383 ping
# 1367 1429 tcp

# chosen_pairs = [
#     ("starlink_550_isls", 1367, 1383, "TcpHybla", full_satellite_network_isls),
#     ("starlink_550_isls", 1367, 1429, "TcpHybla", full_satellite_network_isls)
# ]

# new starlink 550
# 修改：重命名原来的文件夹防止覆盖，修改新的节点对和上面的路径
# Chosen pairs:
# > Itaboraí(133) to Kaunas(116)            1296 + 9 = 1305     1296 + 6 = 1302
# > Itaboraí(133) to Kaunas(116)            2592 + 133 = 2725     2592 + 116 = 2708
# chosen_pairs = [
#     ("new_starlink_550_isls", 2725, 2708, "TcpHybla", full_satellite_network_isls)
# ]

# chosen_pairs = [
#     ("new_starlink_550_isls", 2724, 2729, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2676, 2725, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2724, 2744, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2671, 2744, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2725, 2708, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2660, 2671, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2686, 2725, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2712, 2744, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2712, 2739, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2712, 2729, "TcpHybla", full_satellite_network_isls),

#     ("new_starlink_550_isls", 2740, 2750, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2660, 2679, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2729, 2744, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2729, 2750, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2708, 2712, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2671, 2740, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2679, 2744, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2679, 2725, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2660, 2725, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2660, 2750, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2686, 2750, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2660, 2729, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2712, 2725, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2712, 2724, "TcpHybla", full_satellite_network_isls),

#     ("new_starlink_550_isls", 2663, 2679, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2660, 2686, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2671, 2725, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2740, 2744, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2725, 2744, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2676, 2744, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2671, 2750, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2663, 2744, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2663, 2725, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2679, 2750, "TcpHybla", full_satellite_network_isls),
#     ("new_starlink_550_isls", 2686, 2744, "TcpHybla", full_satellite_network_isls)
# ]


def get_tcp_run_list():
    run_list = []
    for p in chosen_pairs:
        run_list += [
            {
                "name": p[0] + "_" + str(p[1]) + "_to_" + str(p[2]) + "_with_" + p[3] + "_at_10_Mbps",
                "satellite_network": p[4],
                "dynamic_state": dynamic_state,
                "dynamic_state_update_interval_ns": dynamic_state_update_interval_ns,
                "simulation_end_time_ns": simulation_end_time_ns,
                "data_rate_megabit_per_s": 10.0,    # liu: 这里要怎么改
                "queue_size_pkt": 100,
                "enable_isl_utilization_tracking": enable_isl_utilization_tracking,
                "isl_utilization_tracking_interval_ns": isl_utilization_tracking_interval_ns,
                "from_id": p[1],
                "to_id": p[2],
                "tcp_socket_type": p[3],
            },
        ]

    return run_list


def get_pings_run_list():

    # TCP transport protocol does not matter for the ping run
    reduced_chosen_pairs = []
    for p in chosen_pairs:
        if not (p[0], p[1], p[2], p[4]) in reduced_chosen_pairs:
            reduced_chosen_pairs.append((p[0], p[1], p[2], p[4]))  # Stripped out p[3] = transport protocol

    run_list = []
    for p in reduced_chosen_pairs:
        run_list += [
            {
                "name": p[0] + "_" + str(p[1]) + "_to_" + str(p[2]) + "_pings",
                "satellite_network": p[3],
                "dynamic_state": dynamic_state,
                "dynamic_state_update_interval_ns": dynamic_state_update_interval_ns,
                "simulation_end_time_ns": simulation_end_time_ns,
                "data_rate_megabit_per_s": 10000.0,
                "queue_size_pkt": 100000,
                "enable_isl_utilization_tracking": enable_isl_utilization_tracking,
                "isl_utilization_tracking_interval_ns": isl_utilization_tracking_interval_ns,
                "from_id": p[1],
                "to_id": p[2],
                "pingmesh_interval_ns": pingmesh_interval_ns,
            }
        ]

    return run_list