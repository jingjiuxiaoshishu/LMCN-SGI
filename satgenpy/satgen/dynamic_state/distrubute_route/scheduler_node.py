import math
from scipy.constants import speed_of_light

class Scheduler_node:
    def __init__(self,epoch,time_step, sim_duration):
        '''
        初始节点的时间事件调度器及其参数

        :param epoch: 仿真开始时间，格式为 ephem.Date
        :param time_step: 仿真时隙大小
        :param sim_duration: 仿真持续时间
        '''

        self.epoch = epoch
        self.time_step = time_step
        self.sim_duration = sim_duration
        self.curr_slot = 0
        # 总共仿真的时隙数量
        self.num_time_slot = math.ceil(sim_duration/time_step)
        # 初始化time_event_scheduler 为[[],…,[]]
        self.time_event_scheduler = [[] for _ in range(self.num_time_slot)]
        ' time_event_scheduler 的元素为 func,arg* '

    def get_trans_slot_cost(self, distance):
        time_cost = distance / speed_of_light
        # 从发送到接收消息花费的时隙
        slot_cost = math.ceil(time_cost / self.time_step)
        return slot_cost

    def update_curr_slot(self):
        self.curr_slot = self.curr_slot + 1


