import math
from scipy.constants import speed_of_light

# liu:Scheduler_node——基类
class Scheduler_node:
    def __init__(self,epoch,time_step, sim_duration):
        '''
        初始节点的时间事件调度器及其参数

        :param epoch: 仿真开始时间，格式为 ephem.Date
        :param time_step: 仿真时隙大小
        :param sim_duration: 仿真持续时间
        '''

        self.epoch = epoch
        # 输入单位为 ms，需转化为 s
        self.time_step = time_step / 1000
        self.sim_duration = sim_duration
        self.curr_slot = 0
        # 总共仿真的时隙数量
        self.num_time_slot = math.ceil(sim_duration/self.time_step) # liu: math.ceil()向上取整,返回>=x的最小整数
        # 初始化time_event_scheduler 为[[],…,[]]
        self.time_event_scheduler = [[] for _ in range(self.num_time_slot)]
        # delete(' time_event_scheduler 的元素为 func,arg* ')

    # liu:计算转发传输消息花费的时隙数
    def get_trans_slot_cost(self, distance):
        time_cost = distance / speed_of_light
        # 从发送到接收消息花费的时隙
        slot_cost = math.ceil(time_cost / self.time_step)
        return slot_cost

    # liu:更新时隙+1
    def update_curr_slot(self):
        self.curr_slot = self.curr_slot + 1


