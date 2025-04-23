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
from Network import DAC, Value
from Learner import Train
from Dynamic import VehicleDynamics
from Evaluation import evaluation
import matplotlib.pyplot as plt
from math import pi
# from Experiment import evaluation


def built_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--code_mode', default='evaluate', help='train or evaluate')
    parser.add_argument('--evaluate_iteration', default=10000, help='which net to use when evaluate')
    parser.add_argument('--method_version', default='13', help='method_version')

    """task"""
    parser.add_argument('--state_dim', default=4, help='dimension of state')
    parser.add_argument('--action_dim', default=2, help='dimension of action')
    parser.add_argument('--dynamic_dim', default=6, help='dimension of vehicle dynamic')

    """training"""
    parser.add_argument('--buffer_size', default=5000)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--forward_step', default=40)
    parser.add_argument('--gamma', default=0.999)
    parser.add_argument('--lr_p', default=1e-4, help='learning rate of policy network')
    parser.add_argument('--lr_v', default=8e-4, help='learning rate of value network')

    """trajectory"""
    parser.add_argument('--shape', default='dlc2', help='sin, cos, line, traj, dlc or dlc2')
    parser.add_argument('--a', default=0.2, help='amplifier of the sin curve')
    # parser.add_argument('--k', default=1.25*np.pi, help='frequency of the sin curve')
    parser.add_argument('--k', default=1.25, help='frequency of the sin curve')
    parser.add_argument('--y_lim', default=5, help='limitation of y when training')
    parser.add_argument('--psi_lim', default=1.3, help='limitation of psi when training')

    """velocity"""
    parser.add_argument('--target_v', default=0.2, help='default velocity of longitudinal')
    """mode"""
    parser.add_argument('--max_iteration', default=1, help='maximum iteration of training (inner loop)')  # 20000
    parser.add_argument('--max_iteration_out', default=10000, help='maximum iteration of training (outer loop)')
    
    parser.add_argument('--load_data', default=0, help='load pre-trained data for the buffer')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')

    parser.add_argument('--t_steps', default=20, help='diffusion steps')
    parser.add_argument('--al_lr', default=9e-5, help='alpha learning rate')
    parser.add_argument('--lambda_', default=0.999, help='noise control parameter')
    return parser.parse_args()


def main():
    # logging.basicConfig(filename='adp.log', filemode='w',
    #     format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    #     level=logging.INFO)  # 输出运行日志

    torch.manual_seed(1)    # 设置torch在CPU生成随机数的种子
    torch.cuda.manual_seed_all(1)  # 设置torch在GPU生成随机数的种子
    np.random.seed(1)
    args = built_parser()

    device = torch.device(args.device)
    policy = DAC(args).to(device)
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
        # with open(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pkl'),'rb') as f:
        #     policy = pickle.load(f)
        # policy.actor.load_parameters()
        policy = torch.load(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pth'))
        policy.actor.to('cuda:0')
        evaluation(args, policy, dynamic)


def draw():
    with open('../car.json','r',encoding='utf-8') as f:
        data = json.load(f)
    u = []
    v = []
    u1 = data[0]['action']['u'] * 180 / pi
    v1 = data[0]['action']['v'] 
    u.append(u1)
    v.append(-v1)
    count = 0
    for step_data in data:
        if count==0:
            count+=1
            continue
        u2 = step_data['action']['u'] 
        v2 = -step_data['action']['v']
        
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
    for i in range(len(u)):
        u[i] = u[i]* 180 / pi

    plt.rcParams['font.family'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplot(211)
    plt.ylabel(r"$\varphi$(rad)")
    plt.plot(range(300), u, color='mediumblue', linewidth=1, label='输入转向角')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', linewidth=0.5)

    plt.subplot(212)
    plt.ylabel("y(m)")
    plt.plot(range(300), v, color='mediumblue', linewidth=1, label='输入纵向速度')
    plt.legend(loc="upper right")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    main()
    # draw()