import copy
import ephem
from  scheduler_node import Scheduler_node

from  scheduler_node import Scheduler_node
class Sat_selector(Scheduler_node):
    def __init__(self,visible_times,num_orbs,num_sats_per_orbs,shift_between_last_and_first,epoch,time_step, sim_duration):
        super().__init__(epoch,time_step, sim_duration)
        self.visible_times = visible_times
        self.num_orbs = num_orbs
        self.num_sats_per_orbs = num_sats_per_orbs
        self.shift_between_last_and_first = shift_between_last_and_first
        self.num_sats = self.num_orbs * self.num_sats_per_orbs
        self.num_gs = len(visible_times)
        self.gsls = {}
        for gid in range(self.num_gs):
            self.gsls[gid] = [-1, -1]



    # 用于判断链路是否有效，若有，返回预计有效时间
    def check_gsl(self,gid,sid):
        pass

    def classificate(self, sids_visible):
        unclassified_sat_index = {}
        group_1 = []
        group_2 = []
        orbs_of_group_1 = {}
        for sid in sids_visible:
            unclassified_sat_index[sid]= 1

        if not unclassified_sat_index:
            return [], []

        orbs_of_sids_visible = [False for _ in range(self.num_orbs)]
        for sid in sids_visible:
            x, _ = self.get_index_from_sid(sid)
            orbs_of_sids_visible[x] = True

        i = 0
        while True:
            if orbs_of_sids_visible[i]:
                for j in range(3):
                    orbs_of_group_1[(i+j) % self.num_orbs] = 1
                if i in orbs_of_group_1:
                    i = (i+1) % self.num_orbs
                else:
                    break
            else:
                if orbs_of_group_1:
                    break
                else:
                    i = i = (i+1) % self.num_orbs

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


        for sid in sids_visible:
            x, _ = self.get_index_from_sid(sid)
            if x in orbs_of_group_1:
                group_1.append(sid)
                del unclassified_sat_index[sid]

        if unclassified_sat_index:
            group_2 = list(unclassified_sat_index.keys())
        return group_1,group_2

    def get_sats_visible(self,gid):
        sats_visible = {}
        now = ephem.Date(self.epoch + self.curr_slot * self.time_step * ephem.second)
        for sid,visible_time in self.visible_times[gid].items():
            if now >= visible_time[0] and now <= visible_time[1]:
                visible_time_left = (visible_time[1] - now)/ephem.second
                slot_visible_time_left = round(visible_time_left/self.time_step)
                sats_visible[sid] = slot_visible_time_left
        return sats_visible

    def get_sid_with_longest_visible_time_in_group(self,group,sats_visible):
        longest_visible_time = 0
        sid_with_longest_visible_time = -1
        for sid in group:
            if sats_visible[sid] > longest_visible_time:
                sid_with_longest_visible_time = sid
        return sid_with_longest_visible_time

    def update_sats_a_gs_linked_to(self,gid,gsl_to_update):
        sats_visible = self.get_sats_visible(gid)
        # 若无可见卫星，直接返回 -1
        if not sats_visible:
            return -1,-1

        # 获取不更新的 gsl 的 index
        gsl_keep = 1 - gsl_to_update

        group_1,group_2 = self.classificate( list(sats_visible.keys()) )\
        # 根据 group_2 是否非空分情况处理
        if group_2:
            if self.gsls[gid][gsl_keep] == -1:
                sid_1 = sid_linked_to = self.get_sid_with_longest_visible_time_in_group(group_1,sats_visible)
                sid_2 = sid_linked_to = self.get_sid_with_longest_visible_time_in_group(group_2, sats_visible)
                if sats_visible[sid_1] < sats_visible[sid_1]:
                    sid_linked_to = sid_2
                else:
                    sid_linked_to = sid_1
            elif self.gsls[gid][gsl_keep] in group_1:
                sid_linked_to = self.get_sid_with_longest_visible_time_in_group(group_2,sats_visible)
            else:
                sid_linked_to = self.get_sid_with_longest_visible_time_in_group(group_1, sats_visible)
        else:
            if self.gsls[gid][gsl_keep] == -1:
                sid_linked_to = self.get_sid_with_longest_visible_time_in_group(group_1, sats_visible)
            else:
                temp = copy.deepcopy(group_1)
                temp.remove(self.gsls[gid][gsl_keep])
                sid_linked_to = self.get_sid_with_longest_visible_time_in_group(temp, sats_visible)

        self.gsls[gid][gsl_to_update] = sid_linked_to

        if sid_linked_to == -1:
            return -1,-1
        else:
            return sid_linked_to,sats_visible[sid_linked_to]


    def get_index_from_sid(self,sid):
        orb_id = sid // self.num_sats_per_orbs
        sat_id_in_orb = sid % self.num_sats_per_orbs
        return orb_id,sat_id_in_orb

    def get_sid_from_index(self,orb_id,sat_id_in_orb):
        return orb_id*self.num_sats_per_orbs+sat_id_in_orb

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

    def add_gsl_update_event(self,gid,gsl_to_update,slot_to_update_gsl):
        if slot_to_update_gsl < self.num_time_slot:
            msg ={
                "type": "gsl_update",
                "gid": gid,
                "gsl_to_update": gsl_to_update
            }
            self.time_event_scheduler[slot_to_update_gsl].append(msg)

    def deal_msg(self,msg):
        if msg["type"] == "gsl_update":
            gid = msg["gid"]
            gsl_to_update = msg["gsl_to_update"]
            sid,visible_slot_left = self.update_sats_a_gs_linked_to(gid,gsl_to_update)
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

