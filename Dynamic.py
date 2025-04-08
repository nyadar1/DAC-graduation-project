import torch,math
import numpy as np
import csv
import matplotlib.pyplot as plt
from Trajectory import get_traj


class DynamicsConfig():
    L = 0.2358  # wheel base(m)
    m_f = 1.1561  # mass_front(kg)
    m_r = 1.4512  # mass_rear(kg)
    m = m_f + m_r  # mass(kg)  2.6073
    a = L * m_r / m  # distance c.g.to front axle(m) 0.131244
    b = L * m_f / m  # distance c.g.to rear axle(m)  0.104556
    I_zz = m / 3 / L * (a ** 3 + b ** 3)  # yaw moment of inertia(kg * m ^ 2) 2420.0
    # print('m = ', m)  # 2.6073
    # print('a = ', a)  # 0.131244
    # print('b = ', b)  # 0.104556
    # print('I_zz = ', I_zz)  # 0.0120006873
    # C = 1.43        # parameter in Pacejka tire model
    # B = 14.         # parameter in Pacejka tire model
    # u = 15          # longitudinal velocity(m / s)  # todo:最后的跟踪误差一定跟纵向速度有关
    # g = 9.81
    # D = 0.75
    # k1 = 88000      # front axle cornering stiffness for linear model (N / rad)
    # k2 = 94000      # rear axle cornering stiffness for linear model (N / rad)
    # Is = 1.0        # steering ratio
    Ts = 1.0 / 20.0  # control signal period # 60
    # N = 314         # total simulation steps

    """Linear"""
    C_f = 0.31  # 60000
    C_r = 0.58  # 40000
    i_s = 1

    """Force"""
    # F_z1 = m * g * b / L  # Vertical force on front axle
    # F_z2 = m * g * a / L  # Vertical force on rear axle

    """Reference: curve shape of a * sin(kx)"""
    # k_curve = 1 / 30   # 1/30
    # a_curve = 8
    # psi_init = a_curve * k_curve

    """state range"""
    # y_range = 5
    # psi_range = 1.3
    # beta_range = 1.0


class VehicleDynamics(DynamicsConfig):
    def __init__(self, args):
        super(VehicleDynamics, self).__init__()
        self.args = args
        self.device = torch.device(args.device)

        if args.shape == 'traj':
            # read trajectory data
            # test
            data = []

            with open('traj.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    d = [float(row[0]), float(row[1]), float(row[2])]
                    data.append(d)

            self.traj_data = np.array(data)
        # print(self.traj_data.shape)

        self.last_i = 0

        # 这里的_state和init_state是代表的是y和psi的差值
        # [0] <torch, 256 * 6> # [y, v, psi, omega_r, u, x] # TODO ???
        self._state = torch.zeros([self.args.batch_size, self.args.dynamic_dim]).to(self.device)
        self.former_longitudinal_v = torch.zeros([self.args.batch_size, 1]).to(self.device)
        self.former_longitudinal_v_list = [self.former_longitudinal_v]
        self.former_former_v = torch.zeros_like(self.former_longitudinal_v)
        self.count=0
        # [0] <torch, 256 * 6>
        self.init_state = torch.zeros([self.args.batch_size, self.args.dynamic_dim]).to(self.device)

        # init self.init_state <torch, 256 * 6>
        self.initialize_state()

        # [0] <torch, 256 * 1>
        self._reset_index = torch.zeros([self.args.batch_size, 1]).to(self.device)

    def initialize_state(self):
        '''
        random initialization of state
        return:
            init_state <torch, 256 * 6> # abs states
        '''
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        # todo: change the init state
        # ~ N (0.0, 0.6) <torch, 256> # TODO ???
        y = self.init_state[:, 0] = torch.normal(0.0, 0.6, [self.args.batch_size, ]).to(self.device)  # - 2398.6
        # y = self.init_state[:, 0] = torch.from_numpy(np.array([0.6]))

        # ~ N (0.0, 0.4) <torch, 256>
        v = self.init_state[:, 1] = torch.normal(0.0, 0.4, [self.args.batch_size, ]).to(self.device)
        # v = self.init_state[:, 1] = torch.from_numpy(np.array([0.4]))

        # ~ N (0.0, 0.15) <torch, 256>
        psi = self.init_state[:, 2] = torch.normal(0.0, 0.15, [self.args.batch_size, ]).to(self.device)  # - 0.05213
        # psi = self.init_state[:, 2] = torch.from_numpy(np.array([0.15]))

        # ~ N (0.0, 0.1) <torch, 256>
        omega_r = self.init_state[:, 3] = torch.normal(0.0, 0.1, [self.args.batch_size, ]).to(self.device)
        # omega_r = self.init_state[:, 3] = torch.from_numpy(np.array([0.1]))

        # ~ N (0.0, 0.15) <torch, 256>
        beta = torch.normal(0.0, 0.15, [self.args.batch_size]).to(self.device)
        # beta = torch.from_numpy(np.array([0.15]))

        # ~ U (5, 10) <torch, 256>
        u = self.init_state[:, 4] = torch.tensor(np.random.uniform(0, self.args.target_v, [self.args.batch_size])).to(self.device)
        # u = self.init_state[:, 4] = torch.from_numpy(np.array([5]))

        # [0:pi] <torch, 256>
        self.init_state[:, 5] = torch.linspace(0.0, np.pi, self.args.batch_size).to(self.device)  # - 6084.2

        # <torch, 256 * 5>
        init_ref,target_lv = self.ref_traj(self.init_state[:, -1])
        init_ref.to(self.device)
        # <torch, 256 * 6> # col x = 0
        init_ref_all = torch.cat((init_ref, torch.zeros([self.args.batch_size, 1]).to(self.device)), 1)
        init_ref_all = torch.cat((init_ref_all, torch.zeros([self.args.batch_size, 1]).to(self.device)), 1)

        # <torch, 256 * 6> TODO ???
        self._state = self.init_state

        # <torch, 256 * 6>
        init_state = self.init_state + init_ref_all  # 这里的状态包含的是y和psi的绝对值
        # 不太理解init_state为什么要加上init_ref_all
        # <torch, 256 * 6>
        return init_state

    def _reset(self, state):
        '''
        reset the done state to the init_state (abs)
        params:
            state: <torch, 256 * 6>
        return:
            state: <torch, 256 * 6>
        '''
        # for i = 0 : 256
        for i in range(self.args.batch_size):
            if self._reset_index[i] == 1:
                # <torch, 6>
                state[i, :] = self.init_state[i, :]# 怎么不恢复_reset_index呢

        # <torch, 256 * 6>
        return state

    def check_done(self, state):
        '''
        check whether current state exceed the limits
        params:
            state: <torch, 256 * 6>
        return:
            reset_state <torch, 256 * 6>
        '''

        # [256 * 1] Kron [1 * 2] <np, 256 * 2>
        threshold = np.kron(np.ones([self.args.batch_size, 1]),
                            np.array([self.args.y_lim, self.args.psi_lim])).astype('float32')
        # <torch, 256 * 2>
        threshold = torch.from_numpy(threshold).to(self.device)
        # (clone)<torch(detach), 256 * 2> # [y, psi]
        check_state = state[:, [0, 2]].clone().detach_()

        # <torch, 256 * 2> # sign >0:1 <0:-1 =0:0
        sign_error = torch.sign(torch.abs(check_state) - threshold)

        # <torch, 256> # max d1:arr d2:dim
        self._reset_index, _ = torch.max(sign_error, 1)

        # <torch, 256 * 6>
        reset_state = self._reset(state)

        # <torch, 256 * 6>
        return reset_state

    def ref_traj_before(self, x_coordinate):
        '''
        obtain the reference trajectory y_ref & psi_ref in abs coordinate system
        params:
            x_coordinate <torch, 256>
        return:
            state_ref <torch, 256 * 4>
        '''
        
        if self.args.shape == 'sin':
            k, a = self.args.k, self.args.a
            # x_coordinate = x
            width = 1
            straight = 2
            line1 = 0
            line2 = line1 + 1
            line3 = line2 + straight
            line4 = line3 + 1
            cycle = line4 + straight
            xc = x_coordinate - line1
            lane_position = torch.zeros([len(x_coordinate), ]).to(self.device)
            lane_angle = torch.zeros([len(x_coordinate), ]).to(self.device)
            for i in range(len(x_coordinate)):
                if xc[i] < 0:
                    lane_position[i] = 0
                    lane_angle[i] = 0
                else:
                    lane_position[i] = a * torch.sin(k * xc[i]).to(self.device)
                    lane_angle[i] = torch.atan(a * k * torch.cos(k * xc[i])).to(self.device)
            y_ref = lane_position
            psi_ref = lane_angle
            # [0] <torch, 256>
            zeros = torch.zeros([len(x_coordinate)]).to(self.device)
            # print(zeros)
            # [y_ref, 0, psi_ref, 0, 0] <torch, 4 * 256>
            state_ref = torch.cat((y_ref.unsqueeze(0), zeros.unsqueeze(0), psi_ref.unsqueeze(0), zeros.unsqueeze(0)), 0)

        elif self.args.shape == 'line':
            y_ref = torch.zeros([len(x_coordinate)]).to(self.device)
            psi_ref = torch.zeros([len(x_coordinate)]).to(self.device)
            zeros = torch.zeros([len(x_coordinate)]).to(self.device)
            state_ref = torch.cat((y_ref.unsqueeze(0), zeros.unsqueeze(0), psi_ref.unsqueeze(0), zeros.unsqueeze(0)), 0)
        elif self.args.shape == 'traj':
            length = self.traj_data.shape[0]
            zeros = torch.zeros([len(x_coordinate)]).to(self.device)

            # xx = x_coordinate.detach().numpy()
            # print('xx.shape = ', xx.shape)

            y_ref, psi_ref = get_traj(x_coordinate.detach().cpu().numpy()[0])

            # y_ref = torch.from_numpy(y_ref)
            # psi_ref = torch.from_numpy(psi_ref)

            # print('y_ref.shape = ', y_ref.shape)
            # print('psi_ref.shape = ', psi_ref.shape)

            y_ref = torch.from_numpy(np.array([y_ref], dtype=np.float32)).to(self.device)
            psi_ref = torch.from_numpy(np.array([psi_ref], dtype=np.float32)).to(self.device)

            state_ref = torch.cat((y_ref.unsqueeze(0), zeros.unsqueeze(0), psi_ref.unsqueeze(0), zeros.unsqueeze(0)), 0)

            # print(state_ref)
        elif self.args.shape == 'dlc2':
            # x_coordinate = x
            target_lv = torch.zeros([len(x_coordinate), ]).to(self.device)
            width = 0.2
            straight = 2
            line1 = 1.0
            line2 = line1 + 1
            line3 = line2 + straight
            line4 = line3 + 1
            cycle = line4 + 1.0
            xc = x_coordinate % cycle

            max_speed = 0.8
            min_speed = 0.3
            lane_position = torch.zeros([len(x_coordinate), ]).to(self.device)
            lane_angle = torch.zeros([len(x_coordinate), ]).to(self.device)
            
            for i in range(len(x_coordinate)):
                if xc[i] <= line1:
                    lane_position[i] = 0
                    lane_angle[i] = 0
                    
                    if xc[i]>line1-0.5:
                        target_lv[i] = max_speed - (xc[i]-(line1-0.5)) \
                                                *(max_speed-min_speed)/0.5
                    else:
                        target_lv[i] = max_speed

                elif line1 < xc[i] and xc[i] <= line2:
                    lane_position[i] = width * (
                                1 - torch.sin((xc[i] - line1) * torch.pi / (line2 - line1) + torch.pi / 2)) / 2
                    lane_angle[i] = -torch.arctan(width * torch.pi / (line2 - line1) * torch.cos(
                        (xc[i] - line1) * torch.pi / (line2 - line1) + torch.pi / 2) / 2)
                    target_lv[i] = min_speed

                elif line2 < xc[i] and xc[i] <= line3:
                    lane_position[i] = width
                    lane_angle[i] = 0

                    if xc[i]<line2+0.5:
                        target_lv[i] = min_speed + (xc[i]-line2) \
                                                *(max_speed-min_speed)/0.5
                    elif xc[i]>line3-0.5:
                        target_lv[i] = max_speed - (xc[i]-(line3-0.5)) \
                                                *(max_speed-min_speed)/0.5
                    else:
                        target_lv[i] = max_speed

                elif line3 < xc[i] and xc[i] <= line4:
                    lane_position[i] = width * (
                                1 - torch.sin((xc[i] - line3) * torch.pi / (line4 - line3) - torch.pi / 2)) / 2
                    lane_angle[i] = -torch.arctan(width * torch.pi / (line4 - line3) * torch.cos(
                        (xc[i] - line3) * torch.pi / (line4 - line3) - torch.pi / 2) / 2)
                    target_lv[i] = min_speed
                else:
                    lane_position[i] = 0.
                    lane_angle[i] = 0.
                    
                    if xc[i]<line4+0.5:
                        target_lv[i] = min_speed + (xc[i]-line4) \
                                                *(max_speed-min_speed)/0.5
                    else:
                        target_lv[i] = max_speed

            y_ref = lane_position
            # print('y_ref.shape = ', y_ref.shape)
            psi_ref = lane_angle

            zeros = torch.zeros([len(x_coordinate)]).to(self.device)

            state_ref = torch.cat((y_ref.unsqueeze(0), zeros.unsqueeze(0), psi_ref.unsqueeze(0), zeros.unsqueeze(0)), 0)


        elif self.args.shape == 'dlc':
            width = 0.2
            straight = 1
            line1 = 0.5
            line2 = line1 + 1
            line3 = line2 + straight
            line4 = line3 + 1

            cycle = line4 + straight
            x = x_coordinate % cycle
            lane_position = torch.zeros([len(x_coordinate), ]).to(self.device)
            lane_angle = torch.zeros([len(x_coordinate), ]).to(self.device)
            for i in range(len(x_coordinate)):
                if x[i] <= line1:
                    lane_position[i] = 0
                    lane_angle[i] = 0
                elif line1 < x[i] and x[i] <= line2:
                    lane_position[i] = width / (line2 - line1) * (x[i] - line1)
                    lane_angle[i] = np.arctan(width / (line2 - line1))
                elif line2 < x[i] and x[i] <= line3:
                    lane_position[i] = width
                    lane_angle[i] = 0
                elif line3 < x[i] and x[i] <= line4:
                    lane_position[i] = -width / (line4 - line3) * (x[i] - line4)
                    lane_angle[i] = -np.arctan(width / (line4 - line3))
                else:
                    lane_position[i] = 0.
                    lane_angle[i] = 0.

            y_ref = lane_position
            psi_ref = lane_angle
            zeros = torch.zeros([len(x_coordinate)]).to(self.device)

            state_ref = torch.cat((y_ref.unsqueeze(0), zeros.unsqueeze(0), psi_ref.unsqueeze(0), zeros.unsqueeze(0)), 0)

        else:
            print('ERROR!')
            exit(1)

        # <torch, 256 * 4>
        return state_ref.T, target_lv
    # 输入量均为世界坐标系下
    def _state_function(self, state, action, longitudinal_v):
        '''
        state transformation function for both relative and absolute coordinate system
        params:
            state <torch, 256 * 6>
            action <torch, 256 * 2>
        return:
            deriv_state <torch, 256 * 6>
            F_y1 <torch, 256>
            F_y2 <torch, 256>
            alpha1 <torch, 256>
            alpha2 <torch, 256>
        '''
        # state variable
        # <torch, 256>
        v = state[:, 1]

        # <torch, 256>
        psi = state[:, 2]

        # <torch, 256>
        omega_r = state[:, 3]

        # <torch, 256>
        u = state[:, 4]

        # input
        # <torch, 256>
        delta = action[:, 0] #就是控制量u
        delta.requires_grad_(True)

        # derivate of state
        # <torch, 256>
        deriv_v = -2 * (self.C_f + self.C_r) / self.m / u * v \
                  - (2 * (self.a * self.C_f - self.b * self.C_r) / self.m / u + u) * omega_r \
                  + 2 * self.C_f / self.i_s / self.m * delta # + acc[:, 0]*torch.sin(v/u)

        # <torch, 256>
        deriv_omegra_r = -2 * (self.a * self.C_f - self.b * self.C_r) / self.I_zz / u * v \
                         - 2 * (self.a ** 2 * self.C_f + self.b ** 2 * self.C_r) / self.I_zz / u * omega_r \
                         + 2 * self.a * self.C_f / self.i_s / self.I_zz * delta

        # <torch, 256>
        deriv_y = u * torch.sin(psi) + v * torch.cos(psi)
        # <torch, 256>
        deriv_psi = omega_r
        # <torch, 256>
        deriv_u = torch.zeros_like(v * omega_r)
        # deriv_u = acc[:, 0]*torch.cos(v/u)
        # <torch, 256>
        deriv_x = u * torch.cos(psi) - v * torch.sin(psi)

        # <torch, 6 * 256>  各状态量导数
        deriv_state = torch.cat((deriv_y.unsqueeze(0), deriv_v.unsqueeze(0), deriv_psi.unsqueeze(0),
                                 deriv_omegra_r.unsqueeze(0), deriv_u.unsqueeze(0), deriv_x.unsqueeze(0)), 0)

        # <torch, 256 * 6> <256> <256> <256> <256>
        return deriv_state.T# 转置

    def _utility(self, state, control, longitudinal_v):
        '''
        obtain the cost
        params:
            state <torch, 256 * 6>
            control <torch, 256 * 2>
        return:
            utility <torch, 256>
        '''
        # print('input = ',state[:, 2].mean().item(), control[:, 0].mean().item(), longitudinal_v[:, 0].mean().item())
        # <torch, 256>
        utility = 10 * torch.pow(state[:, 0], 2) + 0.5 * torch.pow(state[:, 2], 2) + \
                  0.01 * torch.pow(control[:, 0], 2)
        '''
        V *u(标量)
        '''
        # steer ratio and penalty steer范围为(-1,1)因此ratio直接沿用control[:,0]
        steer_ratio = torch.abs(control[:,0])
        # speed_penalty = 0.5*torch.exp(-torch.abs(longitudinal_v[:,0]))# *torch.pow(longitudinal_v[:,0],2)
        # smooth_v = 0.005*torch.pow(longitudinal_v[:,0]-self.former_longitudinal_v[:,0],2)
        # method 4仅使用speed_penalty = -0.005*torch.pow(longitudinal_v[:,0],2)作为补充
        # speed_penalty += 0.01*torch.exp(2.0*steer_ratio)*torch.pow(longitudinal_v[:,0],2)
        # max_speed = 0.9
        # safe_speed = max_speed*torch.sqrt(1.0-steer_ratio**0.075)
        # speed_constraint = 5*torch.pow(longitudinal_v[:,0]-safe_speed,2)
        # utility += speed_penalty #+ smooth_v
        return utility
    
    def ref_traj(self,x_coordinate):
        if self.args.shape == 'l_dic2':
            # x_coordinate = torch.from_numpy(np.linspace(0,80,4000)).cuda()
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

            xc = x_coordinate % cycle

            lane_position = torch.zeros_like(x_coordinate)
            lane_angle = torch.zeros_like(x_coordinate)

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
            psi_ref = lane_angle
            zeros = torch.zeros([len(x_coordinate)]).to(self.device)
            state_ref = torch.cat((y_ref.unsqueeze(0), zeros.unsqueeze(0), psi_ref.unsqueeze(0), zeros.unsqueeze(0)), 0)
            target_lv = torch.zeros_like(x_coordinate)
            # plt.subplot(211)
            # plt.plot(x_coordinate.cpu().numpy(),y_ref.cpu().numpy())
            # plt.subplot(212)
            # plt.plot(x_coordinate.cpu().numpy(),psi_ref.cpu().numpy())
            # plt.show()
            # input()
        if self.args.shape == 'dlc2':
            # x_coordinate = current_x + deriv_x*self.Ts
            width = 0.2 
            straight = 2
            line1 = 1.0
            line2 = line1 + 1
            line3 = line2 + straight
            line4 = line3 + 1
            cycle = line4 + 1.0
            xc = x_coordinate % cycle

            max_speed = 0.8
            min_speed = 0.3
            delta_speed = max_speed - min_speed
            target_lv = torch.zeros_like(x_coordinate)
            lane_position = torch.zeros_like(x_coordinate)
            lane_angle = torch.zeros_like(x_coordinate)

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

            y_ref = lane_position
            psi_ref = lane_angle
            zeros = torch.zeros([len(x_coordinate)]).to(self.device)
            state_ref = torch.cat((y_ref.unsqueeze(0), zeros.unsqueeze(0), psi_ref.unsqueeze(0), zeros.unsqueeze(0)), 0)

        return state_ref.T, target_lv

    # 这里step需要输入agent state，即各变量均在世界坐标系下
    def step(self, state, action, longitudinal_v,need_utility=False):
        '''
        step forward abs states
        params:
            state <torch, 256 * 6>
            action <torch, 256 * 2>
        return:
            state_next <torch, 256 * 6>
            f_xu <torch, 256 * 6>
            utility <torch, 256>
            F_y1 <torch, 256>
            F_y2 <torch, 256>
            alpha1 <torch, 256>
            alpha2 <torch, 256>
        '''
        deriv_state = self._state_function(state, action, longitudinal_v)
        
        # <torch, 256 * 6> # forward Euler method
        state_next = state + self.Ts * deriv_state

        # state_next[:, 4] = torch.ones_like(state_next[:, 4]) * self.args.target_v
        state_next[:, 4] = longitudinal_v[:, 0]
        # <torch, 256 * 4> # except x
        f_xu = deriv_state[:, 0:4]
        
        # <torch, 256>
        if need_utility:
            utility = self._utility(state, action, longitudinal_v)
            self.former_longitudinal_v = longitudinal_v.clone().detach()
            self.former_longitudinal_v_list.append(self.former_longitudinal_v)
                
        else:
            utility = 0.
            self.former_longitudinal_v_list = [longitudinal_v]
            self.count = 0
        
        # <torch, 256 * 6> <256> <256> <256> <256> <256> <256>
        return state_next, f_xu, utility

    def step_relative(self, state, u, longitudinal_v):
        """
        step forward relative states
        params:
            state <torch, 256 * 6>  传入的agent_batch
            u <torch, 256 * 6>      policy网络传回的控制策略
        return:
            state_next (clone)<torch(detach), 256 * 6>
            state_r_next (clone)<torch(detach), 256 * 6>
        """
        # <torch, 256>
        x_ref, target_lv = self.ref_traj(state[:, -1])

        # (clone)<torch(detach), 256 * 6>
        state_r = state.detach().clone()  # relative state

        # <torch, 256 * 4>
        state_r[:, 0:4] = state[:, 0:4] - x_ref
        state_r[:, 0] = state_r[:, 0] * torch.cos(x_ref[:, 2])

        # <torch, 256 * 6>
        state_next, _, _ = self.step(state, u, longitudinal_v)

        # <torch, 256 * 6>
        state_r_next_bias, _, _ = self.step(state_r, u, longitudinal_v)  # update by relative value

        # <torch, 256 * 6>
        state_r_next = state_r_next_bias.detach().clone()

        # <torch, 256 * 2>
        state_r_next_bias[:, [0, 2]] = state_next[:, [0, 2]]  # y psi with reference update by absolute value

        # <torch, 256>
        x_ref_next,target_lv_next = self.ref_traj(state_next[:, -1])

        # <torch, 256 * 4>
        state_r_next[:, 0:4] = state_r_next_bias[:, 0:4] - x_ref_next

        # (clone)<torch(detach), 256 * 6> (clone)<torch(detach), 256 * 6>
        return state_next.clone().detach(), state_r_next.clone().detach()