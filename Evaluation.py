import time,math

import pandas as pd

from Dynamic import VehicleDynamics
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from Trajectory import get_traj

class Controller:
    def __init__(self):
        # 小车参数
        self.a = 0.131244  # distance c.g.to front axle(m)
        self.L = 0.2358  # wheel base(m)
        self.b = self.L - self.a  # distance c.g.to rear axle(m)
        self.m = 2.6073  # mass(kg)
        self.I_zz = self.m / 3 / self.L * (self.a ** 3 + self.b ** 3)  # yaw moment of inertia(kg * m ^ 2)
        self.u = 0.2  # longitudinal velocity(m / s)  # todo:最后的跟踪误差一定跟纵向速度有关
        self.Is = 1  # steering ratio
        self.Ts = 1/20  # control signal period
        self.N = 40  # total simulation steps
        self.C_f = 0.31
        self.C_r = 0.58

        self.Ac = np.array([[0,1,self.u,0],
                            [0,-2 * (self.C_f + self.C_r) / (self.m * self.u),0,-2 * (self.a * self.C_f - self.b * self.C_r) / (self.m * self.u) + self.u],
                            [0,0,0,1],
                            [0,-2 * (self.a * self.C_f - self.b * self.C_r) / (self.I_zz * self.u),0,-2 * (self.a * self.a * self.C_f + self.b * self.b * self.C_r) / (self.I_zz * self.u)]])

        self.Bc = np.array([0,2 * self.C_f / (self.Is * self.m),
                           0,2 * self.a * self.C_f / (self.Is * self.I_zz)])
        self.C = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        # print(self.Ac)
        self.A=self.Ts*self.Ac+np.eye(4)
        self.B=self.Ts*self.Bc

        self.Q=np.array([[10,0],
                         [0,0.5]])

        self.CA=self.A
        self.CB=self.B
        # self.CE=self.E
        self.CC=np.array([[1,0,0,0],
                          [0,1,0,0]])
        self.CR = 0.1
        self.Cfai = np.zeros((2 * self.N, 4))
        self.Comiga = np.zeros((2 * self.N, self.N))
        self.Comega = np.zeros((2 * self.N, self.N))
        self.CQ_hat = np.zeros((2 * self.N, 2 * self.N))
        self.CR_hat = 0.01*np.eye(self.N)

        for i in range(self.N):
            self.Cfai[2 * i:2 * i + 2, :] = np.matmul(self.CC, np.linalg.matrix_power(self.CA, i + 1))
        for i in range(self.N):
            for j in range(self.N):
                if i < j:
                    self.Comiga[2 * i:2 * i + 2, j] = np.array([0, 0])
                else:
                    self.Comiga[2 * i:2 * i + 2, j] = np.matmul(np.matmul(self.CC, np.linalg.matrix_power(self.CA, i - j)),
                                                               self.CB)
        for i in range(self.N):
            self.CQ_hat[2 * i:2 * i + 2, 2 * i:2 * i + 2] = self.Q
        self.CK_before = np.vstack((np.matmul(np.sqrt(self.CQ_hat), self.Comiga), np.sqrt(self.CR_hat)))
        self.CK = np.matmul(np.linalg.pinv(self.CK_before),
                           np.vstack((np.sqrt(self.CQ_hat), np.zeros((self.N, 2 * self.N)))))

    def update(self,x):
        u=np.matmul(self.CK,-np.matmul(self.Cfai,x.cpu().numpy().reshape(4,1)))[0,0]
        return u
def evaluation(args, policy, dynamic):

    print('Evaluating...')
    device=torch.device(args.device)
    # trajectory
    if args.shape == 'sin':
        state = torch.tensor([[0.0, 0.0, float(np.arctan(args.a * args.k)), 0.0, args.target_v, 0.0]]).to(device)   # sin
    # state = torch.tensor([[-2399.82, 0.0, 0.0, 0.0, 10.0, -1650.46]])     # traj_test
    elif args.shape == 'line':
        state = torch.tensor([[0.5, 0.0, 0.0, 0.0, args.target_v, 0.0]]).to(device)     # line
    else:
        state = torch.tensor([[0.0, 0.0, 0.0, 0.0, args.target_v, 0.0]]).to(device)     # traj

    x_ref, _ = dynamic.ref_traj(state[:, -1])  # x_ref = [y_ref, 0, psi_ref, 0, 0]
    state_r = state.detach().clone()
    state_r[:, 0:4] = state_r[:, 0:4] - x_ref    # todo: 这是相对于车辆坐标系的第一个初始状态

    state_history = state.detach().cpu().numpy()
    action_history = []
    disturb_history = []
    # plt.ion()
    test_range = 1000
    if args.shape == 'traj':
        test_range = 50000
    # start=time.clock()
    t1 = time.time()
    mpc=Controller()

    update_begin = []
    update_end = []
    longitudinal_vs = []
    target_lvs = []
    for i in range(test_range):
        if i % 1000 == 0:
            print(i)
        # render(args, state)
        # todo: input of policy network must be a error state
        update_begin.append(time.time())
        u,longitudinal_v = policy.select_action(state_r[:, 0:4])
        if longitudinal_v.item()!=0:
            longitudinal_vs.append(longitudinal_v.item())
        # print(longitudinal_v.item())
        # print(i,state_r[:, 0:4], u)
        state, state_r = step_relative(dynamic, state, u, longitudinal_v)
        update_end.append(time.time())

        state_history = np.append(state_history, state.detach().cpu().numpy(), axis=0)
        action_history = np.append(action_history, u.detach().cpu().numpy()[:, 0])
        # target_lv = 
    t2 = time.time()
    longitudinal_vs.append(longitudinal_vs[-1])
    print('average update time:', np.mean(np.array(update_end)-np.array(update_begin)))
    print((t2-t1)/test_range)
    print('done!')

    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(9, 8))
    # plt.subplot(121)
    # plt.title('Trajectory')
    x = state_history[:, -1]
    # print('index x>4:',np.where(x>4))
    # print(x[274],x[275])
    print('time to 4:',275*np.mean(np.array(update_end)-np.array(update_begin)))
    if args.shape == 'sin':
        x_coordinate = x
        width = 1
        straight = 2
        line1 = 0
        line2 = line1 + 1
        line3 = line2 + straight
        line4 = line3 + 1
        cycle = line4 + straight
        xc = x_coordinate -line1
        lane_position = np.zeros([len(x_coordinate), ])
        lane_angle = np.zeros([len(x_coordinate), ])
        for i in range(len(x_coordinate)):
            if xc[i] < 0:
                lane_position[i] = 0
                lane_angle[i] = 0
            else:
                lane_position[i] = args.a * np.sin(args.k * xc[i])
                lane_angle[i] = np.arctan(args.k*args.a * np.cos(args.k * xc[i]))
        y_ref = lane_position
        psi_ref = lane_angle

        # y_ref = args.a * np.sin(args.k * x)
        # psi_ref=args.a * np.cos(args.k * x)

    elif args.shape == 'dlc2':
        x_coordinate = x
        width = 0.2 
        straight = 2
        line1 = 1.0
        line2 = line1 + 1
        line3 = line2 + straight
        line4 = line3 + 1
        cycle = line4 + 1.0
        xc = torch.from_numpy(x_coordinate % cycle)

        max_speed = 0.8
        min_speed = 0.3
        delta_speed = max_speed - min_speed
        target_lv = torch.zeros_like(torch.from_numpy(x_coordinate))
        lane_position = torch.zeros_like(torch.from_numpy(x_coordinate))
        lane_angle = torch.zeros_like(torch.from_numpy(x_coordinate))

        mask_le_line1 = xc<=line1
        lane_position[mask_le_line1] = 0
        lane_angle[mask_le_line1] = 0

        mask_line1_line2 = (line1 < xc) & (xc <= line2)
        lane_position[mask_line1_line2] = width * (
                            1 - torch.sin((xc[mask_line1_line2] - line1) * torch.pi / (line2 - line1) + torch.pi / 2)) / 2
        lane_angle[mask_line1_line2] = -torch.arctan(width * torch.pi / (line2 - line1) * torch.cos(
                    (xc[mask_line1_line2] - line1) * torch.pi / (line2 - line1) + torch.pi / 2) / 2)
        
        mask_line2_line3 = (line2 < xc) & (xc <= line3)
        lane_position[mask_line2_line3] = width
        lane_angle[mask_line2_line3] = 0
        
        mask_line3_line4 = (line3 < xc) & (xc <= line4)
        lane_position[mask_line3_line4] = width * (
                            1 - torch.sin((xc[mask_line3_line4] - line3) * torch.pi / (line4 - line3) - torch.pi / 2)) / 2
        lane_angle[mask_line3_line4] = -torch.arctan(width * torch.pi / (line4 - line3) * torch.cos(
                    (xc[mask_line3_line4] - line3) * torch.pi / (line4 - line3) - torch.pi / 2) / 2)
        
        mask_gt_line4 = xc>line4
        lane_position[mask_gt_line4] = 0.
        lane_angle[mask_gt_line4] = 0.


        mask_slowdown = (xc>(line1-0.5)) & mask_le_line1
        target_lv[mask_slowdown] = max_speed- (xc[mask_slowdown]-(line1-0.5))*delta_speed/0.5
        target_lv[mask_le_line1&~mask_slowdown] = max_speed

        target_lv[mask_line1_line2] = min_speed

        mask_ascend = (xc<(line2+0.5))&mask_line2_line3
        mask_descend = (xc>(line3-0.5))&mask_line2_line3
        mask_flat = mask_line2_line3 & ~mask_ascend & ~mask_descend
        target_lv[mask_ascend] = min_speed + (xc[mask_ascend]-line2)*delta_speed/0.5
        target_lv[mask_descend] = max_speed - (xc[mask_descend]-(line3-0.5))*delta_speed/0.5
        target_lv[mask_flat] = max_speed

        target_lv[mask_line3_line4]=min_speed

        mask_final_ascend = (xc<(line4+0.5))&mask_gt_line4
        target_lv[mask_final_ascend]=min_speed+(xc[mask_final_ascend]-line4)*delta_speed/0.5
        target_lv[mask_gt_line4&~mask_final_ascend]=max_speed


        y_ref = lane_position.numpy()
        print('y_ref.shape = ', y_ref.shape)
        psi_ref = lane_angle.numpy()
        xc = xc.numpy()
        target_lv = target_lv.numpy()

    elif args.shape == 'traj':
        # pass
        y_ref = np.zeros(x.shape, dtype=np.float32)
        last_i = 0
        for j in range(x.shape[0]):
            if j % 100 == 0:
                print(j)

            y_ref[j], _ = get_traj(x[j])

    if args.shape == 'line':
        y_ref = np.zeros(len(x))

    if args.shape == 'l_dic2':
        x_coordinate = x
        x_coordinate = np.linspace(0,80,4000)
        width = 0.2 
        straight = 2.0
        circle_straight = 1.0
        line1 = 2.0 # 2
        line2 = line1 + 1.0 # 3
        line3 = line2 + 1.0
        line4 = line3 + 1.0
        line5 = line4 + 2.0
        line6 = line5 + 1.0
        line7 = line6 + 1.0
        cycle = line7 + 1.0

        xc = torch.from_numpy(x_coordinate % cycle)

        lane_position = torch.zeros_like(torch.from_numpy(x_coordinate))
        lane_angle = torch.zeros_like(torch.from_numpy(x_coordinate))

        mask_le_line1 = xc<=line1
        lane_position[mask_le_line1] = 0
        lane_angle[mask_le_line1] = 0

        mask_line1_line2 = (line1 < xc) & (xc <= line2)
        lane_position[mask_line1_line2] = -torch.sqrt(4/3-(xc[mask_line1_line2]-line1)**2)+2/math.sqrt(3.0)
        lane_angle[mask_line1_line2] = torch.arctan((xc[mask_line1_line2]-line1)/(2/math.sqrt(3.0)-lane_position[mask_line1_line2]))
        
        mask_line2_line3 = (line2 < xc) & (xc <= line3)
        lane_position[mask_line2_line3] = (xc[mask_line2_line3]-line2)*math.sqrt(3.0)+1/math.sqrt(3.0)
        lane_angle[mask_line2_line3] = torch.pi/3
        
        mask_line3_line4 = (line3 < xc) & (xc <= line4)
        lane_position[mask_line3_line4] = torch.sqrt(4/3-((xc[mask_line3_line4]-line3)-1)**2)-1/math.sqrt(3.0)+math.sqrt(3.0)+1/math.sqrt(3.0)
        lane_angle[mask_line3_line4] = torch.arctan(-((xc[mask_line3_line4]-line3)-1)/((lane_position[mask_line3_line4]-(math.sqrt(3.0)+1/math.sqrt(3.0)))+1/math.sqrt(3.0)))
        
        mask_line4_line5 = (line4<xc) & (xc<=line5)
        lane_position[mask_line4_line5] = 2/math.sqrt(3)+math.sqrt(3)
        lane_angle[mask_line4_line5] = 0.

        mask_line5_line6 = (line5<xc) & (xc<=line6)
        lane_position[mask_line5_line6] = 2/math.sqrt(3)+math.sqrt(3)+torch.sqrt(4/3-(xc[mask_line5_line6]-line5)**2)-2/math.sqrt(3.0)
        lane_angle[mask_line5_line6] = torch.arctan(-(xc[mask_line5_line6]-line5)/(2/math.sqrt(3.0)+(lane_position[mask_line5_line6]-(2/math.sqrt(3)+math.sqrt(3)))))

        mask_line6_line7 = (line6<xc) & (xc<=line7)
        lane_position[mask_line6_line7] = -(xc[mask_line6_line7]-line6)*math.sqrt(3.0)+1/math.sqrt(3)+math.sqrt(3)
        lane_angle[mask_line6_line7] = -torch.pi/3

        mask_gt_line8 = line7<xc
        lane_position[mask_gt_line8] = -torch.sqrt(4/3-((xc[mask_gt_line8]-line7)-1)**2)+1/math.sqrt(3)+1/math.sqrt(3)
        lane_angle[mask_gt_line8] = torch.arctan((1-(xc[mask_gt_line8]-line7))/((lane_position[mask_gt_line8]-1/math.sqrt(3.0))-1/math.sqrt(3.0)))
        y_ref = lane_position
        print('y_ref.shape = ', y_ref.shape)
        psi_ref = lane_angle

    if args.shape == 'dlc':
        x_coordinate = x
        width = 0.2
        straight = 1
        line1 = 0.5
        line2 = line1 + 1
        line3 = line2 + straight
        line4 = line3 + 1
        cycle = line4 + straight
        xc = x_coordinate % cycle
        lane_position = np.zeros([len(x_coordinate), ])
        lane_angle = np.zeros([len(x_coordinate), ])
        for i in range(len(x_coordinate)):
            if xc[i] <= line1:
                lane_position[i] = 0
                lane_angle[i] = 0
            elif line1 < xc[i] and xc[i] <= line2:
                lane_position[i] = width / (line2 - line1) * (xc[i] - line1)
                lane_angle[i] = np.arctan(width / (line2 - line1))
            elif line2 < xc[i] and xc[i] <= line3:
                lane_position[i] = width
                lane_angle[i] = 0
            elif line3 < xc[i] and xc[i] <= line4:
                lane_position[i] = -width / (line4 - line3) * (xc[i] - line4)
                lane_angle[i] = -np.arctan(width / (line4 - line3))
            else:
                lane_position[i] = 0.
                lane_angle[i] = 0.

        y_ref = lane_position
        print('y_ref.shape = ', y_ref.shape)
        psi_ref = lane_angle

    y = state_history[:, 0]
    tu = np.append(action_history, 0)
    psi = state_history[:, 2]
    print(lane_angle)
    print(len(x_coordinate),len(lane_position.numpy()),len(lane_angle.numpy()))
    plt.subplot(211)
    plt.plot(x_coordinate,lane_position.numpy())
    plt.subplot(212)
    plt.plot(x_coordinate,lane_angle.numpy())
    plt.show()
    # print('y.shape = ', y.shape)
    # plt.plot(dynamic.traj_data[:, 0], dynamic.traj_data[:, 1])
    plt.subplot(411)
    plt.ylabel("y(m)")
    plt.plot(x, y, color='mediumblue', linewidth=1, label='控制轨迹')
    plt.plot(x, y_ref, color='r', linestyle=':', linewidth=1, label='参考轨迹')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', linewidth=0.5)
    # pd.DataFrame(y,x).to_csv("./x_y.csv")
    # pd.DataFrame(y_ref, x).to_csv("./x_y_ref.csv")

    plt.subplot(412)
    plt.ylabel(r"$\varphi$(rad)")
    plt.plot(x, psi, color='mediumblue', linewidth=1, label='车辆横摆角')
    plt.plot(x, psi_ref, color='r', linewidth=1, linestyle=':', label='参考轨迹角度')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', linewidth=0.5)
    # pd.DataFrame(psi, x).to_csv("./x_psi.csv")
    # pd.DataFrame(psi_ref, x).to_csv("./x_psi_ref.csv")

    plt.subplot(413)
    plt.ylabel("u(rad)")
    plt.plot(x, tu)
    plt.grid(True, linestyle='--', linewidth=0.5)
    # pd.DataFrame(tu, x).to_csv("./x_u.csv")

    plt.subplot(414)
    plt.ylabel("纵向速度(m/s)")
    plt.plot(x, longitudinal_vs,linewidth=1, label='纵向速度')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.xlabel('$x$(m)')

    # 统计定量分析
    print('mean y error=',np.sqrt(np.sum((y - y_ref) * (y - y_ref)) / len(y)))
    print('max y error=',max(y - y_ref))
    psi = state_history[:, 2]
    print('mean psi error',np.sqrt(np.sum((psi - psi_ref) * (psi - psi_ref)) / len(y)))
    print('max psi error=',max(psi - psi_ref))
    plt.show()


def step_relative(statemodel, state, u, longitudinal_v):
    x_ref, _ = statemodel.ref_traj(state[:, -1])
    state_r = state.detach().clone()  # relative state
    state_r[:, 0:4] = state_r[:, 0:4] - x_ref
    state_next, _, _ = statemodel.step(state, u, longitudinal_v)
    state_r_next_bias, _, _ = statemodel.step(state_r, u, longitudinal_v)
    state_r_next = state_r_next_bias.detach().clone()
    state_r_next_bias[:, [0, 2]] = state_next[:, [0, 2]]
    x_ref_next,_  = statemodel.ref_traj(state_next[:, -1])
    state_r_next[:, 0:4] = state_r_next_bias[:, 0:4] - x_ref_next
    return state_next.clone().detach(), state_r_next.clone().detach()