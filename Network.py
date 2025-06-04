import os
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class Value(nn.Module):
    def __init__(self, input_num, output_num, args, lr=0.001):
        '''
        initialization
        params:
            input_num <int, 1>
            output_num <int, 1>
            args <args>
            lr <float, 1>
        '''
        super(Value, self).__init__()
        self.args = args
        self.device = torch.device(self.args.device)
        self.layers = nn.Sequential(
            nn.Linear(input_num, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, output_num),
            nn.ReLU(),
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self._initialize_weights()

        # [0, 0, 0, 0, target_v] <torch, 4>
        self._zero_state = torch.tensor([0.0, 0.0, 0.0, 0.0]).to(self.device)

        # [1, 1, 2.4, 2, 0.2] <torch, 1 * 4>
        self._norm_matrix = torch.tensor([1, 1, 2.4, 2], dtype=torch.float32).to(self.device)

    def _initialize_weights(self):
        '''
        initialize network weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init weights by uniform distribution
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        '''
        forward function
        params:
            x <torch, 256 * 4>
        return:
            x <torch, 256 * 2>
        '''
        # <torch, 256 * 4>
        x = torch.mul(x, self._norm_matrix)# element-wise multiplication

        # <torch, 256 * 2>
        x = self.layers(x)

        # <torch, 256 * 2>
        return x

    def save_parameters(self, log_dir, iteration):
        '''
        save network parameters
        params:
            log_dir <str>
            iteration <int, 1>
        '''
        path = self.args.method_version+'_iter' + str(iteration) + '_value_net.pth'
        torch.save(self.state_dict(), os.path.join(log_dir, path))

    def load_parameters(self, log_dir, iteration):
        '''
        load network parameters
        params:
            log_dir <str>
            iteration <int, 1>
        '''
        path = self.args.method_version+'_iter' + str(iteration) + '_value_net.pth'
        self.load_state_dict(torch.load(os.path.join(log_dir, path)))


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
        self.noise_alpha = torch.tensor(0.27, requires_grad=True, device=config.device)
        # self.noise_alpha = torch.tensor(0.05, requires_grad=True, device=config.device)
        
        self.alpha_optim = torch.optim.Adam([self.noise_alpha], lr=config.al_lr)
        self.target_entropy = float(-config.action_dim)
        
        # 超参数
        # self.gamma = config.gamma
        # self.tau = config.tau
        self.lambda_ = config.lambda_
        # self.t_steps = config.t_steps

    def select_action(self, state, eval=False,return_way = 'split', former_V=None):
        action = self.sample(state, add_noise=not eval, 
                                    alpha_param=self.noise_alpha.detach().item(),
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
        alpha_loss = -(self.noise_alpha * (entropy + self.target_entropy)).mean()

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
                '''后续训练注意这里乘了0.05'''
                a_t = mean + 0.05*torch.sqrt(self.beta[t]) * noise
            else:
                a_t = mean   
        # 最终动作添加自适应噪声
        if add_noise:
            x = torch.randn_like(a_t)
            a_t = a_t + lambda_ * alpha_param * x
        a_tt = a_t.clone()
        a_tt[:,0] = torch.tanh(a_t)[:,0]
        if self.config.clip:
            a_tt[:,1] = torch.clamp(torch.sigmoid(a_t)[:,1]+0.05,
                                min=torch.max(torch.tensor(0.01,device=self.config.device),former_V-0.06),
                                max=former_V+0.06)
        else:
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