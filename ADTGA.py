import math
import os
import random

import PubInter
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84


def gen_dummy(point, theta):    # get a new point
    x = point[0]
    y = point[1]
    delt = 0.00005
    new_x = x + delt * math.cos(theta * math.pi / 180)
    new_y = y + delt * math.sin(theta * math.pi / 180)
    return [new_x, new_y, point[2]]


def cal_tdd(t1, t2):    # get TDD
    T = len(t1)
    dis_all = 0
    for i in range(T):
        dis = geod.Inverse(t1[i][0], t1[i][1], t2[i][0], t2[i][1])
        dis_all = dis_all + dis['s12']
    return dis_all / T


def perturbation(traj):
    re = []
    for i in traj:
        x = i[0] + random.uniform(-0.00003, 0.00003)
        y = i[1] + random.uniform(-0.00003, 0.00003)
        re.append([x, y, i[2]])
    return re


def get_one_traj(real_traj, num):
    dummies = []
    for o in range(1, 36):
        O = o * 10
        dum = []
        for g in real_traj:
            new_g = gen_dummy(g, O)
            dum.append(new_g)
        tdd = cal_tdd(real_traj, dum)

        dummies.append( perturbation(dum) )

    ran = random.randint(0, 34)

    store_p = "E:\\ADTGA_traj\\" + str(num) + ".txt"
    f = open(store_p, 'w')
    for gps in dummies[ran]:
        f.write(str(gps[0]) + "," + str(gps[1]) + "," + str(gps[2]) + "\n")
    f.close()

    print("存储:", num)


all_path = PubInter.get_sub_paths("E:\\walk_traj")
for i, p in enumerate(all_path):
    real_ = PubInter.get_traj_2(p)
    get_one_traj(real_, i)

