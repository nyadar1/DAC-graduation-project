import os
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, input_num, output_num, args, lr=0.001):
        '''
        initialization
        params:
            input_num <int, 1> 状态量
            output_num <int, 1> 输出量
            args <args>
            lr <float, 1>
        '''
        super(Policy, self).__init__()
        self.args = args
        self.device=torch.device(self.args.device)
        # self._output_gain = [math.pi / 9, 2.5]
        self._output_gain = [math.pi / 6, 0]

        self.layers = nn.Sequential(
            nn.Linear(input_num, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, output_num),
            # nn.Tanh()   #调整到(-1,1)间
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self._initialize_weights()

        # [y, v, psi, omega] <torch, 1 * 4>
        self._zero_state = torch.tensor([0.0, 0.0, 0.0, 0.0]).to(self.device)

        # [1, 1, 2.4, 2] <torch, 1 * 4>
        self._norm_matrix = torch.tensor([1, 1, 2.4, 2], dtype=torch.float32).to(self.device)


    def forward(self, x):
        '''
        forward function
        params:
            x <torch, 256 * 4>
        return:
            x <torch, 256 * 1>
        '''
        # print('policy x = ', x)
        # <torch, 256 * 4>
        x = torch.mul(x, self._norm_matrix)
        x = self.layers(x)
        steer = torch.tanh(x[:, 0].unsqueeze(1))
        #longitudinal_v = (torch.nn.ELU()(x[:, 1])+ 1 + self.args.target_v).unsqueeze(1)
        #longitudinal_v = (torch.relu(x[:, 1]) + self.args.target_v).unsqueeze(1)
        longitudinal_v = (torch.sigmoid(x[:, 1])*0.2 + self.args.target_v).unsqueeze(1)
        # longitudinal_v = (x[:, 1] + 2).unsqueeze(1)*self.args.target_v
        steer = torch.mul(torch.tensor(self._output_gain).to(self.device), steer)
        return steer, longitudinal_v

    def _initialize_weights(self):
        '''
        initialize network weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init weights by uniform distribution
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)

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

class DiffusionPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.t_steps = config.t_steps
        
        state_dim = self.config.state_dim
        action_dim = self.config.action_dim
        hidden_dim = 256
        # 时间步嵌入
        self.t_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.Tanh()
        )
        
        # 噪声预测网络
        self.noise_net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 扩散调度参数
        self.beta = torch.linspace(1e-4, 0.02, self.t_steps)# .to(self.config.device)
        self.alpha = 1 - self.beta
        # 在第0个维度上累乘
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.optim = torch.optim.Adam(self.parameters(), lr=config.lr_p)
        self._initialize_weights()

        # 解耦控制头
        self.steer_head = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        
        self.speed_head = nn.Sequential(
            nn.Linear(2, 32),
            nn.LayerNorm(32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 速度-转向耦合层
        self.coupling_layer = nn.Linear(1, 1, bias=False)  # 学习转向对速度的影响


    def forward(self, state, action, t):
        # 时间步嵌入
        t_emb = self.t_encoder((t / self.t_steps).view(-1, 1))
        # 噪声预测
        noise = self.noise_net(torch.cat([state, action, t_emb], dim=-1))
        return noise
    
    def sample(self, state, add_noise=True, alpha_param=1.0, lambda_=0.01):
        # 初始action为纯噪声
        assert not torch.isnan(state).any(), "NaN in diffusion state in!"
        a_t = torch.randn(state.shape[0], self.config.action_dim, device=self.config.device)
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
            mean = (a_t - (self.beta[t]/(torch.sqrt(1-alpha_bar_t)+1e-8))*noise_pred) / (torch.sqrt(alpha_t)+1e-8)
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
        # 生成基础控制量
        steer = self.steer_head(a_t) * 1.0 # 1.0为max steer
        raw_speed = self.speed_head(a_t) * 0.9 # 0.9为max speed
        
        # 应用耦合约束
        steer_impact = self.coupling_layer(torch.abs(steer)/1.0)
        coupled_speed = raw_speed * (1 - steer_impact.sigmoid())

        # a_tt[:,0] = (torch.sigmoid(a_t)[:,0]-0.5)*2
        # a_tt[:,1] = 0.8*torch.sigmoid(a_t)[:,1]+0.1
        return torch.cat([steer, coupled_speed], dim=-1)
    
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
                torch.nn.init.constant_(m.bias, 0.0)

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
        # x = torch.mul(x, self._norm_matrix)# element-wise multiplication

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


# DAC算法主类
class DAC:
    def __init__(self, config):
        
        # 网络初始化
        self.actor = DiffusionPolicy(config).to(config.device)
        
        # 熵参数
        self.alpha = torch.tensor(0.27, requires_grad=True, device=config.device)
        self.alpha_optim = torch.optim.Adam([self.alpha], lr=config.al_lr)
        self.target_entropy = float(-config.action_dim)
        
        # 超参数
        # self.gamma = config.gamma
        # self.tau = config.tau
        self.lambda_ = config.lambda_
        # self.t_steps = config.t_steps

    def select_action(self, state, eval=False,return_way = 'split'):
        action = self.actor.sample(state, add_noise=not eval, 
                                    alpha_param=self.alpha.detach().item(),
                                    lambda_=self.lambda_)
        if return_way == 'split':
            return action[:,0].unsqueeze(1),action[:,1].unsqueeze(1)
        elif return_way == 'merge':
            return action
        else:
            raise NotImplementedError
    
    def policy_entropy_update(self,state,control):
        states = state  # bu使用子样本估计
        with torch.no_grad():
            multiple_actions = [self.select_action(states,return_way='merge') for _ in range(20)]
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
        if math.isinf(entropy.mean().item()) or math.isnan(entropy.mean().item()):
            print(f_hat,entropy,actions)
            input("f_hat corrupted")
    
        return entropy