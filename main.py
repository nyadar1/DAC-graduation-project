#!/usr/bin/python
# -*- coding: utf-8 -*-

# ================================================================
# Author: Geng Jiale
# Time: 2024/07/11
# Function: using Approximate Dynamic Programming (ADP) to train control policy, value function
# ================================================================

import os
import time
import pickle
import torch
import logging
import argparse
import json
import numpy as np
from Network import Value, DAC2
from Learner import Train
from Dynamic import VehicleDynamics
from Evaluation import evaluation
import matplotlib.pyplot as plt
from math import pi
# from Experiment import evaluation


def built_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--code_mode', default='evaluate', help='train or evaluate')
    parser.add_argument('--evaluate_iteration', default=10000,type=int, help='which net to use when evaluate')
    parser.add_argument('--method_version', default='10-alpha02', help='method_version')

    """task"""
    parser.add_argument('--state_dim', default=4, help='dimension of state')
    parser.add_argument('--action_dim', default=2, help='dimension of action')
    parser.add_argument('--dynamic_dim', default=6, help='dimension of vehicle dynamic')

    """training"""
    parser.add_argument('--buffer_size', default=5000)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--forward_step',type=int, default=20)
    parser.add_argument('--gamma', default=0.999)
    parser.add_argument('--lr_p', default=1e-4, help='learning rate of policy network')
    parser.add_argument('--lr_v', default=8e-4, help='learning rate of value network')

    """trajectory"""
    parser.add_argument('--shape', default='dlc2', help='sin, cos, line, traj, dlc or dlc2')
    parser.add_argument('--a', default=0.2, help='amplifier of the sin curve')
    # parser.add_argument('--k', default=1.25*np.pi, help='frequency of the sin curve')
    parser.add_argument('--k', default=np.pi/2, help='frequency of the sin curve')
    parser.add_argument('--y_lim', default=5, help='limitation of y when training')
    parser.add_argument('--psi_lim', default=1.3, help='limitation of psi when training')

    """velocity"""
    parser.add_argument('--target_v', default=0.2, help='default velocity of longitudinal')
    """mode"""
    parser.add_argument('--max_iteration', default=1, help='maximum iteration of training (inner loop)')  # 20000
    parser.add_argument('--max_iteration_out', default=10000, help='maximum iteration of training (outer loop)')
    
    parser.add_argument('--load_data', default=0, help='load pre-trained data for the buffer')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')

    parser.add_argument('--t_steps', default=20, type=int,help='diffusion steps')
    parser.add_argument('--al_lr', default=2e-4, help='alpha learning rate')# 9e-5
    parser.add_argument('--lambda_', default=0.999, help='noise control parameter')

    parser.add_argument('--smooth_v', default=5, type=float,help='noise control parameter')
    parser.add_argument('--enlarge_v', default=1e-4,type=float, help='noise control parameter')
    parser.add_argument('--clip', default=True,type=bool, help='noise control parameter')
    return parser.parse_args()


def main(args):
    # logging.basicConfig(filename='adp.log', filemode='w',
    #     format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    #     level=logging.INFO)  # 输出运行日志

    # torch.manual_seed(1)    # 设置torch在CPU生成随机数的种子
    # torch.cuda.manual_seed_all(1)  # 设置torch在GPU生成随机数的种子
    # np.random.seed(1)
    

    device = torch.device(args.device)
    policy = DAC2(args).to(device)
    value = Value(args.state_dim, 1, args, lr=args.lr_v).to(device)

    dynamic = VehicleDynamics(args)

    policy_log_dir = './weights/policy_net/'
    os.makedirs(policy_log_dir, exist_ok=True)  # 递归创建目录
    value_log_dir = './weights/value_net/'
    os.makedirs(value_log_dir, exist_ok=True)

    if args.code_mode == 'train':
        train = Train(args)
        # logging.debug(train)
        begin = time.time()
        if args.load_data == 1:
            value.load_parameters(value_log_dir, 10000)
            policy.actor.load_parameters(policy_log_dir, 10000)

        train.agent_batch = dynamic.initialize_state()
        for iter_index_out in range(args.max_iteration_out):
            for iter_index in range(args.max_iteration):
                logging.debug(iter_index)

                train.state_update(policy, dynamic)

                value_loss = train.value_update(policy, value, dynamic)
                logging.debug(value_loss)
                logging.debug('==============================')

                policy_loss = train.policy_update(policy, value, iter_index_out)
                logging.debug(policy_loss)

                t_iter_index = iter_index_out + 1
                if t_iter_index % 100 == 0:
                    print('iteration:{:3d} | policy loss:{:.4f} | value loss:{:.4f} '.format(t_iter_index, float(policy_loss), float(value_loss)))
                if t_iter_index % 1000 == 0:
                    now = time.time()
                    print('Using time:',int((now-begin)/60),'min')
                    # with open(os.path.join(policy_log_dir,f'{args.method_version}_DAC{t_iter_index}.pkl'),
                    #             'wb') as f:
                    #     pickle.dump(policy, f)
                    torch.save(policy,os.path.join(policy_log_dir,f'{args.method_version}_DAC{t_iter_index}.pth'))
                    value.save_parameters(value_log_dir, t_iter_index)

        train.plot_figure()

    elif args.code_mode == 'evaluate':
        '''tracking performance depend on the trajectory shape k_curve and longitudinal speed u'''
        if os.path.exists(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pkl')):
            print("Use pickle")

            with open(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pkl'),'rb') as f:
                policy = pickle.load(f)
            # policy.actor.load_parameters()
            evaluation(args, policy, dynamic)
        else:
            print('Use pth')
            policy = torch.load(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pth'),map_location=args.device)
            # policy.config.device='cpu'
            evaluation(args, policy, dynamic)


def draw():
    with open('../小车ADP/car.json','r',encoding='utf-8') as f:
        data = json.load(f)
    u = []
    v = []
    eys = []
    efais = []
    max_error = 0
    u1 = data[0]['action']['u'] * 180 / pi
    v1 = data[0]['action']['v'] 
    u.append(u1)
    v.append(-v1)
    count = 0
    for step_data in data:
        if count==0:
            count+=1
            ey = step_data['state'][0]
            efai = step_data['state'][2]
            eys.append(ey)
            efais.append(efai)
            continue
        u2 = step_data['action']['u'] 
        v2 = -step_data['action']['v']
        ey = step_data['state'][0]
        efai = step_data['state'][2]
        eys.append(ey)
        efais.append(efai)
        if ey>max_error:
            max_error = ey
        if u2-u1>0.03:
            tempu=u1+0.03
            u.append(tempu)
        elif u2-u1<-0.03:
            tempu=u1-0.03
            u.append(tempu)
        else:
            tempu = u2
            u.append(u2)
        u1 = tempu

        if v2-v1>0.06:
            tempv = v1+0.06
            v.append(tempv)
        elif v2-v1<-0.06:
            tempv = v1-0.06
            v.append(tempv)
        else:
            tempv = v2
            v.append(v2)
        v1 = tempv
    # for i in range(len(u)):
    #     u[i] = u[i]* 180 / pi
    print('max error',max_error)
    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplot(411)
    plt.ylabel("y[m]")
    plt.plot(range(300), eys, linewidth=1, label='纵向误差值')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.subplot(412)
    plt.ylabel(r"$\varphi$[rad]")
    plt.plot(range(300), efais, linewidth=1, label='转角误差值')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.subplot(413)
    plt.ylabel(r"$\varphi$[rad]")
    plt.plot(range(300), u, linewidth=1, label='输入转向角')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.subplot(414)
    plt.ylabel("V[m/s]")
    plt.plot(range(300), v, linewidth=1, label='输入纵向速度')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    
    # draw()
    # torch.autograd.set_detect_anomaly(True)
    seed = 1
    args = built_parser()
    torch.manual_seed(seed)    # 设置torch在CPU生成随机数的种子
    torch.cuda.manual_seed_all(seed)  # 设置torch在GPU生成随机数的种子
    np.random.seed(seed)
    if args.code_mode == 'train':
        # 可以考虑增大noise_alpha的学习率，减小noise_lambda
        # 按查重版4.3图开始的顺序配置：
        # 4.3对应18版本，
        '''目前看，sin训练在40步forward下非clip不稳定，dlc2训练在40步forward下非clip稳定'''
        # # 对应图4.3，与论文匹配，就是这组参数
        # args.method_version="0"
        # args.clip=True
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=20
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="13-alpha02"
        # args.clip=False
        # args.shape="sin"
        # args.smooth_v=5
        # args.enlarge_v=1e-5
        # args.forward_step=20
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        '''重新设计，保证平滑'''
        # args.method_version="9-alpha02"
        # args.clip=False
        # args.shape="sin"
        # args.smooth_v=5
        # args.enlarge_v=1e-5
        # args.forward_step=40
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="10-alpha02"
        # args.clip=False
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-5
        # args.forward_step=40
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="10-alpha02-1e-4"
        # args.clip=False
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=40
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="15-alpha005-2e-4-sqrtbeta005"
        # args.clip=False
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-5
        # args.forward_step=40
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="11-alpha02"
        # args.clip=True
        # args.shape="sin"
        # args.smooth_v=5
        # args.enlarge_v=1e-5
        # args.forward_step=40
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="12-alpha02"
        # args.clip=True
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-5
        # args.forward_step=40
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')
    else:
        args.clip=True
        args.shape="dlc2"
        main(args=args)



        # 4.3调整种子

        # args.method_version="0--1alpha005"
        # args.clip=True
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=20
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # # 对应图4.4，与论文中一致，原method中应该是丢失了
        # args.method_version="1"
        # args.clip=True
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=40
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # # 对应图4.5，new中的method18-1
        # args.method_version="2"
        # args.clip=True
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=10
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # # 对应图4.6，new中的method18-2
        # args.method_version="3"
        # args.clip=True
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=20
        # args.t_steps = 10
        # main(args)
        # print(args.method_version,' over')

        # # 对应图4.7，new中的method18-0的9000测试
        # args.method_version="4"
        # args.clip=True
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=20
        # args.t_steps = 30
        # main(args)
        # print(args.method_version,' over')

        # # 对应图4.8，没有问题，new中的method本应是0-1，但是应该被覆盖掉了
        # args.method_version="5"
        # args.clip=False
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=40
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # # new中的method 0 图4.9 这个是最平滑的，在这个上面先调整forward_step。
        # args.method_version="6-0"
        # args.clip=False
        # args.shape="sin"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=20
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # # new中的method 0 图4.9 这个是最平滑的，在这个上面先调整forward_step。
        # args.method_version="6-0-alpha02"
        # args.clip=False
        # args.shape="sin"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=20
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="6-alpha02"
        # args.clip=False
        # args.shape="sin"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=40
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # new中的method 0 图4.9 这个是最平滑的，调整种子试试
        # torch.manual_seed(seed)    # 设置torch在CPU生成随机数的种子
        # torch.cuda.manual_seed_all(seed)  # 设置torch在GPU生成随机数的种子
        # np.random.seed(seed)
        # args.method_version="6--2"
        # args.clip=False
        # args.shape="sin"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=40
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # # new中的method 14-0 图4.10
        # args.method_version="7"
        # args.clip=True
        # args.shape="dlc2"
        # args.smooth_v=20
        # args.enlarge_v=1e-4
        # args.forward_step=20
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # new中的method 16 图4.11
        # args.method_version="8"
        # args.clip=True
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-3
        # args.forward_step=40
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')
        # main(args)