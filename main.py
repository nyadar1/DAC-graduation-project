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
from Network import DAC, Value, DAC2
from Learner import Train
from Dynamic import VehicleDynamics
from Evaluation import evaluation
import matplotlib.pyplot as plt
from math import pi
# from Experiment import evaluation


def built_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--code_mode', default='evaluate', help='train or evaluate')
    parser.add_argument('--evaluate_iteration', default=9000, help='which net to use when evaluate')
    parser.add_argument('--method_version', default='0-1', help='method_version')

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
    parser.add_argument('--al_lr', default=9e-5, help='alpha learning rate')
    parser.add_argument('--lambda_', default=0.999, help='noise control parameter')

    parser.add_argument('--smooth_v', default=5, type=float,help='noise control parameter')
    parser.add_argument('--enlarge_v', default=1e-4,type=float, help='noise control parameter')
    parser.add_argument('--clip', default=True,type=bool, help='noise control parameter')
    return parser.parse_args()


def main(args):
    # logging.basicConfig(filename='adp.log', filemode='w',
    #     format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    #     level=logging.INFO)  # 输出运行日志

    torch.manual_seed(1)    # 设置torch在CPU生成随机数的种子
    torch.cuda.manual_seed_all(1)  # 设置torch在GPU生成随机数的种子
    np.random.seed(1)
    

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
        if os.path.exists(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pkl')):
            print("Use pickle")

            with open(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pkl'),'rb') as f:
                policy = pickle.load(f)
            # policy.actor.load_parameters()
            evaluation(args, policy, dynamic)
        else:
            print('Use pth')
            policy = torch.load(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pth'))
            policy.actor.to('cuda:0')
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

def move_to_cpu(args):
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    os.environ["PYTHON_CUDA_ALLOC_CONF"]="none"
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.is_available = lambda: False
    args.device='cpu'
    policy = DAC(args)
    policy_log_dir = './weights/policy_net/'
    policy = torch.load(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pth'),map_location='cpu')
    torch.save(policy,f'{args.method_version}_DAC10000cpu.pth')

def new_test(args):
    torch.manual_seed(1)    # 设置torch在CPU生成随机数的种子
    torch.cuda.manual_seed_all(1)  # 设置torch在GPU生成随机数的种子
    np.random.seed(1)
    device = 'cpu'
    args.device='cpu'
    policy = DAC(args).to(device)
    policy2 = DAC2(args).to(device)
    value = Value(args.state_dim, 1, args, lr=args.lr_v).to(device)
    dynamic = VehicleDynamics(args)
    policy_log_dir = './weights/policy_net/'
    os.makedirs(policy_log_dir, exist_ok=True)  # 递归创建目录
    value_log_dir = './weights/value_net/'
    os.makedirs(value_log_dir, exist_ok=True)
    if os.path.exists(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pkl')):
        print("Use pickle")
        with open(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pkl'),'rb') as f:
            policy = pickle.load(f)
        # policy.actor.load_parameters()
        evaluation(args, policy, dynamic)
    else:
        print('Use pth')
        policy = torch.load(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pth'))
        copy_params(policy, policy2)
        evaluation(args, policy2, dynamic)


if __name__ == '__main__':
    
    # draw()
    # torch.autograd.set_detect_anomaly(True)
    '''
    18效果最好，以18为核心调整参数
    '''
    # draw()
    args = built_parser()
    # move_to_cpu(args)
    if args.code_mode == 'train':
        # python main.py  --method_version=18-0 --forward_step=20 --t_steps=30
        # python main.py   --method_version=14-0 --forward_step=20 --t_steps=20 --smooth_v=20 --enlarge_v=1e-4 --code_mode=train应当是有clamp的
        
        # 无clamp操作
        # args.method_version="0"
        # args.clip=False
        # args.shape="sin"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=40
        # main(args)
        # print(args.method_version,' over')图4.9

        # args.method_version="0-1"
        # args.clip=False
        # args.shape="dlc2"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=40
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="18-18"
        # args.clip=True
        # args.shape="sin"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=20
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="18-1"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=10
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="18-2"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=20
        # args.t_steps = 10
        # main(args)
        # print(args.method_version,' over')


        # args.al_lr=9e-4
        # args.method_version="18-3"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=20
        # args.t_steps = 20
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="15"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="15-1"
        # args.smooth_v=0
        # args.enlarge_v=0
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="16"
        # args.smooth_v=5
        # args.enlarge_v=1e-3
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="17"
        # args.smooth_v=5
        # args.enlarge_v=1e-2
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="19"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=5
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="20"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=1
        # main(args)
        # print(args.method_version,' over')

        # args.method_version="21"
        # args.smooth_v=5
        # args.enlarge_v=1
        # args.forward_step=40
        # main(args)
        # print(args.method_version,' over')

        # args.t_steps = 10
        # args.method_version="22"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=40
        # main(args)
        # print(args.method_version,' over')

        # args.t_steps = 2
        # args.method_version="23"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=40
        # main(args)
        # print(args.method_version,' over')

        # args.t_steps = 1
        # args.method_version="24"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=40
        # main(args)
        # print(args.method_version,' over')


        # args.t_steps = 20
        # args.method_version="coupled"
        # args.smooth_v=5
        # args.enlarge_v=1e-4
        # args.forward_step=20
        # main(args)
        # print(args.method_version,' over')
        main(args=args)
    else:
        main(args=args)
        # new_test(args)
