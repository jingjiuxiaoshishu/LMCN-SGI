#!/bin/bash

# 实验文件夹列表
experiments=( "global_shortest_path" "maximum_service_time" "correct_modified_routing_system" "modified_global_shortest_path" "routing_system")

for exp_folder in "${experiments[@]}"; do

    target_folder=~/Desktop/hypatia_routing/hypatia/paper/satellite_networks_state/gen_data/starlink_550_isls_plus_grid_ground_stations_starlink_algorithm_free_one_only_over_isls
    
    if [ -d "$target_folder" ]; then
        rm -r "$target_folder"
        echo "已删除目标文件夹 $target_folder"
    fi

    # 复制对应实验的路由表
    cp -r ~/Desktop/result_plot/result_of_all_routing_table/"$exp_folder"/starlink_550_isls_plus_grid_ground_stations_starlink_algorithm_free_one_only_over_isls "$target_folder"
    echo "复制实验 ${exp_folder} 的路由表到 $target_folder"

    # 切换到实验脚本目录
    cd ~/Desktop/hypatia_routing/hypatia/paper/ns3_experiments/a_b/
    echo "切换到实验脚本目录"

    # 调用 Python 脚本 step_1_generate_runs.py
    python step_1_generate_runs.py
    echo "step_1_generate_runs 执行完成"

    # 调用 Python 脚本 step_2_run.py
    python step_2_run.py
    echo "step_2_run 执行完成"

    # 调用 Python 脚本 step_3_generate_plots.py
    python step_3_generate_plots.py
    echo "step_3_generate_plots 执行完成"

    mkdir "${exp_folder}_ns3_results_new_pairs"
    cp -r runs "${exp_folder}_ns3_results_new_pairs"
    cp -r data "${exp_folder}_ns3_results_new_pairs"
    cp -r pdf "${exp_folder}_ns3_results_new_pairs"

    # 打包实验结果
    tar -czvf "${exp_folder}_ns3_results_new_pairs.tar.gz" "${exp_folder}_ns3_results_new_pairs"
    rm -r "${exp_folder}_ns3_results"

    echo "$exp_folder 实验完成并打包结果。"
done
