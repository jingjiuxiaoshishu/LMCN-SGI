import ephem
from mudule_in_hypatia import read_ground_stations_extended
from mudule_in_hypatia import read_tles

class Gs_with_gid():
    def __init__(self, gs):
        self.gid = gs["gid"]
        self.ground_station =ephem.Observer()
        self.ground_station.lat = gs["latitude_degrees_str"]
        self.ground_station.lon = gs["longitude_degrees_str"]


class Satellites_with_sid:
    def __init__(self, sid, satellite):
        self.sid = sid
        self.satellite = satellite


class Visible_time_helper:
    def __init__(self, ground_stations,satellites,min_elevation,epoch,time_step, sim_duration):
        self.epoch =epoch
        self.time_step = time_step
        self.sim_duration = sim_duration
        # super().__init__(epoch,time_step, sim_duration)
        self.min_elevation = min_elevation
        visible_times = {}
        for gid in range(len(ground_stations)):
            print(gid,"的可见卫星计算中")
            visible_times[gid] = {}
            for sid in range(len(satellites)):
                visible_time = self.calculate_visible_time(satellites[sid], ground_stations[gid])
                visible_times[gid][sid] = visible_time
        self.visible_times = visible_times

    def calculate_visible_time(self, satellite, ground_station):
        i = 0
        visible_flag = False
        while i <= self.sim_duration:
            ground_station.date = self.epoch + i * ephem.second
            satellite.compute(ground_station)
            if satellite.alt * 180 / 3.1416 > self.min_elevation:  # 检查卫星的仰角是否大于最小仰角
                visible_flag = True
                start = ground_station.date
                break
            else:
                i = i + self.time_step

        if visible_flag:
            while i <= self.sim_duration:
                ground_station.date = self.epoch + i * ephem.second
                satellite.compute(ground_station)
                if satellite.alt * 180 / 3.1416 > self.min_elevation:  # 检查卫星的仰角是否大于最小仰角
                    end = ground_station.date
                    i = i + self.time_step
                else:
                    break
            return [start, end]
        else:
            return [-1, -1]


if __name__ == "__main__":

    ground_stations_dict = read_ground_stations_extended("ground_stations.txt")

    ground_stations = []
    for ground_station_dict in ground_stations_dict:
        ground_station = ephem.Observer()
        ground_station.lat = ground_station_dict["latitude_degrees_str"]
        ground_station.lon = ground_station_dict["longitude_degrees_str"]
        ground_stations.append(ground_station)


    tles = read_tles("tles.txt")
    satellites = tles["satellites"]


    # 获取历元信息并转换历元为日期时间格式
    epoch = ephem.Date(satellites[1]._epoch)

    # 打印历元信息
    print("Epoch (DateTime):", epoch)
    min_elevation = 25
    sim_duration = 400
    time_step = 0.05

    start = ephem.now()
    print(start)

    visible_time_helper = Visible_time_helper(ground_stations, satellites, min_elevation, epoch,
                                              time_step, sim_duration)
    import pickle
    data = [1, 2, 3, 4, 5]
    # with open('visible_times.pkl', 'wb') as f:
    #     pickle.dump(visible_time_helper.visible_times, f)

    stop = ephem.now()
    print((stop - start))

