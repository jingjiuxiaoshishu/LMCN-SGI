import copy
import ephem
from  .scheduler_node import Scheduler_node

# liu: add comments
class Sat_selector(Scheduler_node):
    def __init__(self,visible_times,num_orbs,num_sats_per_orbs,shift_between_last_and_first,epoch,time_step, sim_duration):
        super().__init__(epoch,time_step, sim_duration)
        self.visible_times = visible_times  # 二维字典，第一维为 gid，第二维为 sid，值为可见时间段
        self.num_orbs = num_orbs    # 卫星轨道数
        self.num_sats_per_orbs = num_sats_per_orbs  # 每个轨道上的卫星数
        self.shift_between_last_and_first = shift_between_last_and_first    # 最后一个轨道与第一个轨道的偏移
        self.num_sats = self.num_orbs * self.num_sats_per_orbs  # 卫星总数
        self.num_gs = len(visible_times)    # len(visible_times)获得gid的个数
        self.gsls = {}  # key为 gid，值为 gsl
        for gid in range(self.num_gs):
            self.gsls[gid] = [-1, -1]   # TODO:两条GSL？



    # 用于判断链路是否有效，若有，返回预计有效时间
    def check_gsl(self,gid,sid):
        pass

    def classificate(self, sids_visible):
        unclassified_sat_index = {}
        group_1 = []
        group_2 = []
        orbs_of_group_1 = {}
        for sid in sids_visible:    # liu:将所有可见卫星的id设置为未分类
            unclassified_sat_index[sid]= 1

        if not unclassified_sat_index:  # liu:没有可见卫星，返回空列表
            return [], []

        orbs_of_sids_visible = [False for _ in range(self.num_orbs)]    # liu:标记可见卫星所在的轨道列表
        for sid in sids_visible:
            x, _ = self.get_index_from_sid(sid)
            orbs_of_sids_visible[x] = True

        i = 0   # liu:轨道索引
        while True:
            if orbs_of_sids_visible[i]:
                for j in range(3):  # liu:从0~2的循环，用于设置orbs_of_group_1中3个连续键i,i+1,i+2的值为1
                    orbs_of_group_1[(i+j) % self.num_orbs] = 1  
                if i in orbs_of_group_1:    # liu:如果i在字典中，则自增1
                    i = (i+1) % self.num_orbs
                else:   # liu:如果i不在字典中，则跳出循环
                    break
            else:   # liu:orbs_of_sids_visible[i]为false
                if orbs_of_group_1: # liu:orbs_of_group_1不为空,则跳出循环
                    break
                else:   # liu:orbs_of_group_1为空,则自增1
                    i = i = (i+1) % self.num_orbs
        # liu:作用
        # 将与orbs_of_sids_visible[i]相关的卫星轨道（连续的3个）标记到orbs_of_group_1中。
        # 当i在orbs_of_sids_visible中不再是True，orbs_of_group_1不为空时，停止标记过程。

        # liu:TODO 这个for循环是干啥的
        for i in  range(3):
            if i in orbs_of_group_1:
                while True:
                    if orbs_of_sids_visible[i]:
                        for j in range(3):
                            orbs_of_group_1[(i - j) % self.num_orbs] = 1
                    if i in orbs_of_group_1:
                        i = (i - 1) % self.num_orbs
                    else:
                        i = (i - 1) % self.num_orbs
                        break
                break

        # liu:根据其所在轨道的分类，每个卫星ID被分类到 group_1 或留在 unclassified_sat_index 中。
        # 所有仍然在 unclassified_sat_index 中的卫星ID都被归入 group_2
        for sid in sids_visible:
            x, _ = self.get_index_from_sid(sid)
            if x in orbs_of_group_1:
                group_1.append(sid)
                del unclassified_sat_index[sid]

        if unclassified_sat_index:
            group_2 = list(unclassified_sat_index.keys())
        return group_1,group_2

    # liu:获得gid对应地面站的可见卫星，返回sats_visible，key为sid，值为剩余可见时隙
    def get_sats_visible(self,gid):
        sats_visible = {}
        now = ephem.Date(self.epoch + self.curr_slot * self.time_step * ephem.second)   # 当前时间
        for sid,visible_time in self.visible_times[gid].items():
            if now >= visible_time[0] and now <= visible_time[1]:
                visible_time_left = (visible_time[1] - now)/ephem.second    # liu:剩余可见时长
                slot_visible_time_left = round(visible_time_left/self.time_step)    # liu:剩余可见时隙
                # 默认 + 1  个时隙，以便处理只有一个时隙可见的情况
                sats_visible[sid] = slot_visible_time_left + 1
        return sats_visible

    # liu:获取分组内的剩余可见时间最长的卫星的sid
    def get_sid_with_longest_visible_time_in_group(self,group,sats_visible):
        longest_visible_time = 0
        sid_with_longest_visible_time = -1
        for sid in group:
            if sats_visible[sid] > longest_visible_time:
                sid_with_longest_visible_time = sid
        return sid_with_longest_visible_time

    # 更新一个gs链接的gsl
    def update_sats_a_gs_linked_to(self,gid,gsl_to_update):
        sats_visible = self.get_sats_visible(gid)   # liu:获得gid对应地面站的可见卫星，key为sid，值为剩余可见时隙
        # 若无可见卫星，直接返回 -1
        if not sats_visible:
            return -1,-1

        # 获取不更新的 gsl 的 index
        gsl_keep = 1 - gsl_to_update    # gsl_to_update为0或1

        group_1,group_2 = self.classificate( list(sats_visible.keys()) )
        # 根据 group_2 是否非空分情况处理
        if group_2: # liu:group_2非空
            if self.gsls[gid][gsl_keep] == -1:  # liu:如果不更新的gsl为-1(TODO:不存在？)，则更新sid_linked_to为group_1和group_2中剩余可见时间较长的卫星
                sid_1 = sid_linked_to = self.get_sid_with_longest_visible_time_in_group(group_1,sats_visible)
                sid_2 = sid_linked_to = self.get_sid_with_longest_visible_time_in_group(group_2, sats_visible)
                if sats_visible[sid_1] < sats_visible[sid_1]:
                    sid_linked_to = sid_2
                else:
                    sid_linked_to = sid_1
            elif self.gsls[gid][gsl_keep] in group_1:   # liu:如果不更新的gsl在group_1中，则更新sid_linked_to为group_2中剩余可见时间最长的卫星
                sid_linked_to = self.get_sid_with_longest_visible_time_in_group(group_2,sats_visible)
            else:   # liu:如果不更新的gsl在group_2中，则更新sid_linked_to为group_1中剩余可见时间最长的卫星
                sid_linked_to = self.get_sid_with_longest_visible_time_in_group(group_1, sats_visible)
        else:   # liu:group_2为空
            if self.gsls[gid][gsl_keep] == -1:  # liu:如果不更新的gsl为-1(TODO:不存在？)，则更新sid_linked_to为group_1中剩余可见时间最长的卫星
                sid_linked_to = self.get_sid_with_longest_visible_time_in_group(group_1, sats_visible)
            else:   # liu:如果不更新的gsl不为-1，则更新sid_linked_to为group_1中剩余可见时间最长的卫星，但是要排除不更新的gsl
                temp = copy.deepcopy(group_1)
                temp.remove(self.gsls[gid][gsl_keep])
                sid_linked_to = self.get_sid_with_longest_visible_time_in_group(temp, sats_visible)

        self.gsls[gid][gsl_to_update] = sid_linked_to   # liu:更新gsl_to_update为sid_linked_to

        if sid_linked_to == -1:
            return -1,-1
        else:
            return sid_linked_to,sats_visible[sid_linked_to]    # 返回链接的卫星的sid，以及剩余可见时隙

    # liu:获取sid卫星对应的轨道号和轨道内卫星号
    def get_index_from_sid(self,sid):
        orb_id = sid // self.num_sats_per_orbs
        sat_id_in_orb = sid % self.num_sats_per_orbs
        return orb_id,sat_id_in_orb

    # liu:获取轨道号和轨道内卫星号对应的sid
    def get_sid_from_index(self,orb_id,sat_id_in_orb):
        return orb_id*self.num_sats_per_orbs+sat_id_in_orb

    # liu:TODO
    def get_neighbor_index(self,sid,x,y):
        orb_id, sat_id_in_orb = self.get_index_from_sid(sid)
        neighbor_orb_id = orb_id+x
        if neighbor_orb_id >= self.num_orbs:
            neighbor_orb_id = neighbor_orb_id - self.num_orbs
            neighbor_sat_id_in_orb = (sat_id_in_orb + y + self.shift_between_last_and_first)%self.num_sats_per_orbs
        elif neighbor_orb_id < 0:
            neighbor_orb_id = neighbor_orb_id + self.num_orbs
            neighbor_sat_id_in_orb = (sat_id_in_orb + y - self.shift_between_last_and_first) % self.num_sats_per_orbs
        else:
            neighbor_sat_id_in_orb = (sat_id_in_orb + y) % self.num_sats_per_orbs
        return neighbor_orb_id,neighbor_sat_id_in_orb

    # liu:添加gsl更新事件
    def add_gsl_update_event(self,gid,gsl_to_update,slot_to_update_gsl):
        if slot_to_update_gsl < self.num_time_slot:
            msg ={
                "type": "gsl_update",
                "gid": gid,
                "gsl_to_update": gsl_to_update
            }
            self.time_event_scheduler[slot_to_update_gsl].append(msg)   # liu:添加消息到时间事件调度器

    # liu:处理消息
    def deal_msg(self,msg):
        if msg["type"] == "gsl_update":
            gid = msg["gid"]
            gsl_to_update = msg["gsl_to_update"]
            sid,visible_slot_left = self.update_sats_a_gs_linked_to(gid,gsl_to_update)  # 更新gsl
            if sid!= -1:
                self.add_gsl_update_event(gid,gsl_to_update,self.curr_slot+visible_slot_left)
            else:
                self.add_gsl_update_event(gid,gsl_to_update,self.curr_slot+1)

    def process(self):
        for msg in self.time_event_scheduler[self.curr_slot]:
            self.deal_msg(msg)
        self.curr_slot = self.curr_slot + 1


if __name__ == "__main__":
    import pickle
    with open('visible_times.pkl', 'rb') as f:
        visible_times = pickle.load(f)
    num_orbs = 72
    num_sats_per_orbs = 18
    shift_between_last_and_first = 8
    epoch = ephem.Date("2000/0/0 00:00:00")
    time_step = 0.1
    sim_duration = 400
    sat_selector = Sat_selector(visible_times,num_orbs,num_sats_per_orbs,shift_between_last_and_first,epoch,time_step, sim_duration)
    for gid,gsl in sat_selector.gsls.items():
        # 对每个 gs 的 gsl 更新两次
        for i in range(2):
            sat_selector.add_gsl_update_event(gid,i,0)
    for i in range(sat_selector.num_time_slot):
        print(i)
        sat_selector.process()
        if i==0:
            with open("gsls.pkl","wb") as f:
                pickle.dump(sat_selector.gsls, f)


    sum = 0
    for i in range(sat_selector.num_time_slot):
        if i >1000 and i<3000:
            for msg in sat_selector.time_event_scheduler[i]:
                sum = sum + 1

