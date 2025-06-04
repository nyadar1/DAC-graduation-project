#!/usr/bin/python
# -*- coding: utf-8 -*-

# ================================================================
# Author: Geng Jiale
# Time: 2024/07/08
# Function: evaluate ADP when training
# ================================================================

import os
import torch
import logging
import argparse
import numpy as np
from Network import Policy, Value
from Learner import Train
from Dynamic import VehicleDynamics
from Evaluation import evaluation
# from Experiment import evaluation


def built_parser():
    parser = argparse.ArgumentParser()

    """task"""
    parser.add_argument('--state_dim', default=4, help='dimension of state')
    parser.add_argument('--action_dim', default=1, help='dimension of action')
    parser.add_argument('--dynamic_dim', default=6, help='dimension of vehicle dynamic')

    """training"""
    parser.add_argument('--buffer_size', default=5000)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--forward_step', default=40)
    parser.add_argument('--gamma', default=0.999)
    parser.add_argument('--lr_p', default=3e-5, help='learning rate of policy network')
    parser.add_argument('--lr_v', default=8e-4, help='learning rate of value network')

    """trajectory"""
    parser.add_argument('--shape', default='dlc2', help='sin, cos, line, traj, dlc or dlc2')
    parser.add_argument('--a', default=0.2, help='amplifier of the sin curve')
    parser.add_argument('--k', default=1.5, help='frequency of the sin curve')
    parser.add_argument('--y_lim', default=5, help='learning rate of value network')
    parser.add_argument('--psi_lim', default=1.3, help='learning rate of value network')

    """velocity"""
    parser.add_argument('--target_v', default=0.2, help='default velocity of longitudinal')
    """mode"""
    parser.add_argument('--max_iteration', default=1, help='maximum iteration of training (inner loop)')  # 20000
    parser.add_argument('--max_iteration_out', default=10000, help='maximum iteration of training (outer loop)')
    parser.add_argument('--code_mode', default='evaluate', help='train or evaluate')
    parser.add_argument('--evaluate_iteration', default=6000, help='which net to use when evaluate')
    parser.add_argument('--load_data', default=0, help='load pre-trained data for the buffer')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
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
    policy = Policy(args.state_dim, args.action_dim, args, lr=args.lr_p).to(device)
    value = Value(args.state_dim, 1, args, lr=args.lr_v).to(device)

    dynamic = VehicleDynamics(args)

    policy_log_dir = './data_without_value/policy_net/'
    os.makedirs(policy_log_dir, exist_ok=True)  # 递归创建目录
    value_log_dir = './data/value_net/'
    os.makedirs(value_log_dir, exist_ok=True)

    if args.code_mode == 'train':
        train = Train(args)
        # logging.debug(train)

        if args.load_data == 1:
            value.load_parameters(value_log_dir, 10000)
            policy.load_parameters(policy_log_dir, 10000)

        train.agent_batch = dynamic.initialize_state()
        for iter_index_out in range(args.max_iteration_out):
            for iter_index in range(args.max_iteration):
                logging.debug(iter_index)

                train.state_update(policy, dynamic)

                value_loss = train.value_update(policy, value, dynamic)
                logging.debug(value_loss)
                logging.debug('==============================')

                policy_loss = train.policy_update(policy, value)
                logging.debug(policy_loss)

                t_iter_index = iter_index + 1
                if t_iter_index % 1 == 0:
                    print('iteration:{:3d} | policy loss:{:.4f} | value loss:{:.4f} '.format(t_iter_index, float(policy_loss), float(value_loss)))
                if t_iter_index % args.max_iteration == 0:
                    value.save_parameters(value_log_dir, t_iter_index+iter_index_out)
                    policy.save_parameters(policy_log_dir, t_iter_index+iter_index_out)

        train.plot_figure()

    elif args.code_mode == 'evaluate':
        '''tracking performance depend on the trajectory shape k_curve and longitudinal speed u'''
        policy.load_parameters(policy_log_dir, args.evaluate_iteration)
        evaluation(args, policy, dynamic)


if __name__ == '__main__':
    main()