import os
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
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

class DAC2(nn.Module):
    def __init__(self, config):
        super(DAC2, self).__init__()
        # 网络初始化
        self.config = config
        self.t_steps = config.t_steps
        
        state_dim = self.config.state_dim
        action_dim = self.config.action_dim
        hidden_dim = 256
        # 时间步嵌入
        self.t_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.Mish(),
            nn.Linear(16, 16)
        )
        
        # 噪声预测网络
        self.noise_net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 16, hidden_dim),
            nn.Mish(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )
        # self.coupled = nn.Linear(1, 1,bias=False)
        
        # 扩散调度参数
        self.beta = torch.linspace(1e-4, 0.02, self.t_steps)# .to(self.config.device)
        self.alpha = 1 - self.beta
        # 在第0个维度上累乘
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.optim = torch.optim.Adam(self.parameters(), lr=config.lr_p)
        self._initialize_weights()
        
        # 熵参数
        self.alpha_noise = torch.tensor(0.2616, requires_grad=True, device=config.device)
        self.alpha_optim = torch.optim.Adam([self.alpha_noise], lr=config.al_lr)
        self.target_entropy = float(-config.action_dim)
        
        # 超参数
        # self.gamma = config.gamma
        # self.tau = config.tau
        self.lambda_ = config.lambda_
        # self.t_steps = config.t_steps

    def select_action(self, state, eval=False,return_way = 'split', former_V=None):
        action = self.sample(state, add_noise=not eval, 
                                    alpha_param=self.alpha_noise.detach().item(),
                                    lambda_=self.lambda_,
                                    former_V=former_V)
        if return_way == 'split':
            return action[:,0].unsqueeze(1),action[:,1].unsqueeze(1)
        elif return_way == 'merge':
            return action
        else:
            raise NotImplementedError
    
    def policy_entropy_update(self,state,control):
        states = state  # bu使用子样本估计
        with torch.no_grad():
            multiple_actions = [self.select_action(states,return_way='merge',former_V=control[:,1]) for _ in range(20)]
            # multiple_actions = [self.actor.sample(states) for _ in range(20)]
            actions = [control for _ in range(20)]
        
        # KDE估计熵
        entropy = self.kernel_density_estimate(multiple_actions,actions)
        alpha_loss = -(self.alpha * (entropy + self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        return alpha_loss

    def kernel_density_estimate(self, sample_actions, actions, bandwidth=0.1):
        """
        高斯核密度估计的对数概率计算
        sample_actions为state经过diffusion采样20遍的动作:[...],一个元素为一遍,共20个元素
        action为下一步要采用的动作。shape:[b,2], 重复20遍, 也为列表
        """
        K = []
        for sa,a in zip(sample_actions,actions):
             K.append(torch.exp(-(sa-a)**2/(2*bandwidth**2)))
        f_hat = 1/(bandwidth*np.sqrt(2 * np.pi))*(sum(K)/len(K))+1e-8
        entropy = torch.log(f_hat)
        assert not torch.isnan(entropy).any(), "NaN in entropy!"
        return entropy
        
    def forward(self, state, action, t):
        # 时间步嵌入
        t_emb = self.t_encoder((t / self.t_steps).view(-1, 1))
        # 噪声预测
        noise = self.noise_net(torch.cat([state, action, t_emb], dim=-1))
        return noise
    
    def sample(self, state, add_noise=True, alpha_param=1.0, lambda_=0.01, former_V=None):
        # 初始action为纯噪声
        a_t = torch.randn(state.shape[0], self.config.action_dim, device=self.config.device)
        assert not torch.isnan(state).any(), "NaN in diffusion state in!"
        # 反向扩散过程
        for t in reversed(range(self.t_steps)):
            # torch.full全为t，形状为(state.shape[0])的张量即shap为[256]，作为时序张量
            t_tensor = torch.full((state.shape[0],), t, device=self.config.device).float()
            '''
            self方法调用nn.Module的__call__方法
            会直接调用forward函数, 同时保证梯度, 钩子等结构的完整
            等价于定义模型model后直接model(y)
            '''
            noise_pred = self(state, a_t, t_tensor)
            assert not torch.isnan(noise_pred).any(), "NaN in noise_pred!"
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            
            # 计算均值
            mean = (a_t - (self.beta[t]/torch.sqrt(1-alpha_bar_t))*noise_pred) / torch.sqrt(alpha_t)
            
            # 添加噪声
            if t > 0 and add_noise:
                noise = torch.randn_like(a_t)
                a_t = mean + torch.sqrt(self.beta[t]) * noise
            else:
                a_t = mean   
        # 最终动作添加自适应噪声
        if add_noise:
            x = torch.randn_like(a_t)
            a_t = a_t + lambda_ * alpha_param * x
        a_tt = a_t.clone()
        a_tt[:,0] = torch.tanh(a_t)[:,0]

        # 处理 NaN 的 former_V
        # former_V_safe = torch.where(
        #     torch.isnan(former_V),
        #     torch.tensor(0.1, device=former_V.device),  # 替换 NaN 为默认值
        #     former_V
        # )
        #if self.config.clip:
        # a_tt[:,1] = torch.clamp(torch.sigmoid(a_t)[:,1]+0.05,
        #                        min=torch.max(torch.tensor(0.01,device=self.config.device),former_V-0.06),
        #                         max=former_V+0.06)
        
        # a_tt[:,1] = (torch.sigmoid(a_t)[:,1]+0.05)*torch.tanh(a_t)[:,0]
        # print('att[:,1]',a_tt[:,1])
        a_tt[:,1] = torch.sigmoid(a_t)[:,1]+0.05
        
        return a_tt
    
    def save_parameters(self, log_dir, iteration):
        '''
        save network parameters
        params:
            log_dir <str>
            iteration <int, 1>
        '''
        path = self.args.method_version+'_iter' + str(iteration) + '_policy_net.pth'
        torch.save(self.state_dict(), os.path.join(log_dir, path))

    def load_parameters(self, log_dir, iteration):
        '''
        load network parameters
        params:
            log_dir <str>
            iteration <int, 1>
        '''
        path = self.args.method_version+'_iter' + str(iteration) + '_policy_net.pth'
        self.load_state_dict(torch.load(os.path.join(log_dir, path)))

    def _initialize_weights(self):
        '''
        initialize network weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init weights by uniform distribution
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

def copy_params(src_model, dst_model, name_mapping=None):
    src_dict = src_model.state_dict()
    print(src_dict.keys())
    dst_dict = dst_model.state_dict()
    src_keys = [
    'actor.t_encoder.0.weight', 'actor.t_encoder.0.bias',
    'actor.t_encoder.2.weight', 'actor.t_encoder.2.bias',
    'actor.noise_net.0.weight', 'actor.noise_net.0.bias',
    'actor.noise_net.2.weight', 'actor.noise_net.2.bias',
    'actor.noise_net.3.weight', 'actor.noise_net.3.bias',
    'actor.noise_net.5.weight', 'actor.noise_net.5.bias',
    'actor.noise_net.6.weight', 'actor.noise_net.6.bias'
    ]

    # 自动生成 name_mapping（移除 actor. 前缀）
    name_mapping = {key: key.replace('actor.', '', 1) for key in src_keys}
    # name_mapping['alpha', 'alpha_noise']
    # 自动名称映射（如果未提供自定义映射）
    if name_mapping is None:
        name_mapping = {k: k for k in src_dict.keys() if k in dst_dict}
    
    # 逐参数拷贝
    for src_name, dst_name in name_mapping.items():
        if src_dict[src_name].shape != dst_dict[dst_name].shape:
            print(f"形状不匹配: {src_name}({src_dict[src_name].shape}) vs {dst_name}({dst_dict[dst_name].shape})")
            continue
            
        if isinstance(src_dict[src_name], torch.nn.Parameter):
            data = src_dict[src_name].data
            print(src_name,'copyed to data')
        else:
            data = src_dict[src_name]
            print(src_name,'copyed to data. not Parameter')
        # 执行拷贝
        dst_dict[dst_name].copy_(data)
    
    # 加载修改后的参数
    dst_model.load_state_dict(dst_dict)
    # print(src_model.state_dict(), 'copyed to', dst_model.state_dict())

def built_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--code_mode', default='evaluate', help='train or evaluate')
    parser.add_argument('--evaluate_iteration', default=10000, help='which net to use when evaluate')
    parser.add_argument('--method_version', default='0', help='method_version')

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
    parser.add_argument('--device', default='cuda', help='cuda:0 or cpu')

    parser.add_argument('--t_steps', default=20, type=int,help='diffusion steps')
    parser.add_argument('--al_lr', default=9e-5, help='alpha learning rate')
    parser.add_argument('--lambda_', default=0.999, help='noise control parameter')

    parser.add_argument('--smooth_v', default=5, type=float,help='noise control parameter')
    parser.add_argument('--enlarge_v', default=1e-4,type=float, help='noise control parameter')
    parser.add_argument('--clip', default=True,type=bool, help='noise control parameter')
    return parser.parse_args()

def main(args):
    torch.manual_seed(1)    # 设置torch在CPU生成随机数的种子
    torch.cuda.manual_seed_all(1)  # 设置torch在GPU生成随机数的种子
    np.random.seed(1)
    device = 'cuda'
    policy = DAC(args).to(device)
    args.device='cpu'
    policy2 = DAC2(args).to('cpu')
    dynamic = VehicleDynamics(args)

    policy_log_dir = './weights/policy_net/'
    os.makedirs(policy_log_dir, exist_ok=True)  # 递归创建目录
    value_log_dir = './weights/value_net/'
    os.makedirs(value_log_dir, exist_ok=True)
    if os.path.exists(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pkl')):
        print("Use pickle")

        with open(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pkl'),'rb') as f:
            policy = pickle.load(f)
        evaluation(args, policy, dynamic)
    else:
        print('Use pth')
        policy = torch.load(os.path.join(policy_log_dir,f'{args.method_version}_DAC{args.evaluate_iteration}.pth'), map_location='cuda')
        policy.actor.to('cuda')
        copy_params(policy, policy2)
        print(policy2.alpha_noise)
        evaluation(args, policy2.to('cpu'), dynamic)
        torch.save(policy2, os.path.join(policy_log_dir,f'cpu_{args.method_version}_DAC{args.evaluate_iteration}.pth'))

if __name__ == '__main__':
    args = built_parser()
    main(args)

