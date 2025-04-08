#!/usr/bin/python
# -*- coding: utf-8 -*-

# ================================================================
# Author: Geng Jiale
# Time: 2024/07/11
# Function: using Approximate Dynamic Programming (ADP) to train control policy, value function
# 
# Modified by Cheng JunJie
# Time: 2025/03
# Function: mainly for diffusion policy and longitudinal velocity
# ================================================================

import os
import torch
import logging
import argparse
import pickle
import time
import numpy as np
from Network import DAC, Value
from Learner import Train
from Dynamic import VehicleDynamics
from Evaluation import evaluation
# from Experiment import evaluation


def built_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--code_mode', default='train', help='train or evaluate')
    parser.add_argument('--evaluate_iteration', default=10000, help='which net to use when evaluate')
    """task"""
    parser.add_argument('--state_dim', default=4, help='dimension of state')
    parser.add_argument('--action_dim', default=2, help='dimension of action')
    parser.add_argument('--dynamic_dim', default=6, help='dimension of vehicle dynamic')
    # method version3开始引入目标纵向速度，目前现在dlc2上训练
    parser.add_argument('--method_version', default='6', help='method_version')
    """training"""
    parser.add_argument('--buffer_size', default=5000)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--forward_step', default=40)
    parser.add_argument('--gamma', default=0.999)
    parser.add_argument('--lr_p', default=1e-4, help='learning rate of policy network')
    parser.add_argument('--lr_v', default=8e-4, help='learning rate of value network')

    """trajectory"""
    # // cSpell:disable-next-line
    parser.add_argument('--shape', default='dlc2', help='square,sin, cos, line, traj, dlc or dlc2')
    parser.add_argument('--a', default=0.2, help='amplifier of the sin curve')
    parser.add_argument('--k', default=1.25*np.pi, help='frequency of the sin curve')
    parser.add_argument('--y_lim', default=5, help='limitation of y when training')
    parser.add_argument('--psi_lim', default=1.3, help='limitation of psi when training')

    """velocity"""
    parser.add_argument('--target_v', default=0.2, help='default velocity of longitudinal')
    """mode"""
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
    policy = DAC(args)
    value = Value(args.state_dim, 1, args, lr=args.lr_v).to(device)

    dynamic = VehicleDynamics(args)

    policy_log_dir = './weights/policy_net/'
    os.makedirs(policy_log_dir, exist_ok=True)  # 递归创建目录
    value_log_dir = './weights/value_net/'
    os.makedirs(value_log_dir, exist_ok=True)
    '''
    由于训练过慢(forward step中需要更新target_lv)
    因此考虑将target_lv加在状态量后面，每次x坐标的新增值指导下一步target_lv
    所以现在的状态量为[y,v(横向),psi,w,V(纵向),x_coordinate]
    仅修改了ref_traj函数,不需要向状态量中添加target_lv,效率很高滴
    不使用参考的纵向速度，而是希望纵向速度尽可能大(在转向角u较小时)
    method 4未参与git，因为只是在loss中添加了
    method 5使用指数最小化纵向速度变化量与最大化其大小
    '''
    if args.code_mode == 'train':
        train = Train(args)
        # logging.debug(train)
        begin = time.time()
        if args.load_data == 1:
            value.load_parameters(value_log_dir, 10000)
            policy.actor.load_parameters(policy_log_dir, 10000)

        train.agent_batch = dynamic.initialize_state()
        for iter_index_out in range(args.max_iteration_out):
            train.state_update(policy, dynamic)
            value_loss = train.value_update(policy, value, dynamic)
            logging.debug(value_loss)
            logging.debug('==============================')

            policy_loss = train.policy_update(policy, value, iter_index_out)
            logging.debug(policy_loss)

            t_iter_index = iter_index_out + 1
            if t_iter_index % 100 == 0:
                print('iteration:{:3d} | policy loss:{:.4f} | value loss:{:.4f} '.format(t_iter_index, float(policy_loss), float(value_loss)))
            # if t_iter_index % 100 == 0:
            #     policy.actor.save_parameters(policy_log_dir, t_iter_index)
            if t_iter_index % 1000 == 0:
                    now = time.time()
                    print('Using time:',int((now-begin)/60),'min')
                    with open(os.path.join(policy_log_dir,f'{args.method_version}_DAC{t_iter_index}.pkl'),
                                'wb') as f:
                        pickle.dump(policy, f)
                    value.save_parameters(value_log_dir, t_iter_index)

        train.plot_figure()

    elif args.code_mode == 'evaluate':

        '''tracking performance depend on the trajectory shape k_curve and longitudinal speed u'''
        with open(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pkl'),'rb') as f:
            policy = pickle.load(f)
        # policy.actor.load_parameters()
        policy.actor.to('cuda:0')
        evaluation(args, policy, dynamic)


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    main()