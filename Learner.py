import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import logging
import pickle
# todo value的forward方式会被我换成__call__方式，policy换成select action方式

class Train(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(self.args.device)

        # [] <torch, 256 * 6>
        self.agent_batch = torch.empty([self.args.batch_size, self.args.dynamic_dim]).to(self.device)  # todo: 绝对值
        # logging.debug(self.agent_batch)

        # [] <torch, 256 * 6>
        self.state_batch = torch.empty([self.args.batch_size, self.args.dynamic_dim]).to(self.device)  # todo: 差值
        # logging.debug(self.state_batch)

        # [1] <torch, 256 * 1>
        self.init_index = np.ones([self.args.batch_size, 1])
        # logging.debug(self.init_index)

        self.x_forward = []
        # logging.debug(self.x_forward)

        self.u_forward = []
        self.longitudinal_v_forward = []
        self.w_forward = []
        # logging.debug(self.u_forward)

        self.l_forward = []
        # logging.debug(self.l_forward)

        # [0, 1] <np, 1 * 2>
        self.value_loss = np.empty([0, 1])
        # logging.debug(self.value_loss)
        self.alphas = []
        # [0, 1] <np, 1 * 2>
        self.policy_loss = np.empty([0, 1])
        # logging.debug(self.policy_loss)

        # for i = 0 : 19
        for i in range(self.args.forward_step):
            self.u_forward.append([])
            self.longitudinal_v_forward.append([])
            self.w_forward.append([])
            self.l_forward.append([])
            # self.u_forward = [] <list, 20>
            # self.l_forward = [] <list, 20>
        # logging.debug(self.u_forward)
        # logging.debug(self.l_forward)

        # for i = 0 : 20
        for i in range(self.args.forward_step + 1):
            self.x_forward.append([])
            # self.x_forward = [] <list, 21>
        # logging.debug(self.x_forward)

    # policy <NN>, dynamics <Dyanmic>
    def state_update(self, policy, dynamics):
        # <torch, 256 * 6>
        self.agent_batch = dynamics.check_done(self.agent_batch)# 检查limits
        self.agent_batch.detach_()
        logging.debug(self.agent_batch)

        # <torch, 256 * 5> abs trajectory  0-3.1416的linspace
        ref_trajectory, target_lv = dynamics.ref_traj(self.agent_batch[:, -1])
        logging.debug(ref_trajectory)

        # <torch, 256 * 5> # relative states
        self.state_batch[:, 0:4] = self.agent_batch[:, 0:4] - ref_trajectory
        self.state_batch.detach()
        logging.debug(self.state_batch)

        # 这里传入的是state_batch, 因为policy网络的输入必须是差值才有意义
        # policy产生控制策略
        # <torch, 256 * 2>
        self.control, self.velocity = policy.select_action(self.state_batch[:, 0:4])
        logging.debug(self.control)

        # 这里传入的是agent_batch而不是state_batch,状态整体到达下一个位置
        # <torch, 256 * 6>, <torch, 256 * 6>
        self.agent_batch, self.state_batch = dynamics.step_relative(self.agent_batch, self.control, self.velocity)
        logging.debug(self.agent_batch)
        logging.debug(self.state_batch)

    # policy <NN>, value <NN>
    def policy_update(self, policy, value, iter_index_out):
        # <torch, 256 * 1>
        # self.value_next = value(self.state_batch_next)
        self.value_next = value(self.state_batch_next)

        # <torch, 1> # J_actor
        policy_loss = torch.mean(self.sum_utility + self.value_next)

        policy.actor.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 3.0)
        policy.actor.optim.step()

        if (iter_index_out+1)%100==0:
            with torch.no_grad():
                self.control, self.velocity = policy.select_action(self.state_batch[:, 0:4])
                
            # policy.policy_entropy_update(self.state_batch_next[:,:4],
            #                              torch.cat([self.control, self.velocity],dim=1))
            policy.policy_entropy_update(self.state_batch_next,
                                          torch.cat([self.control, self.velocity],dim=1))
            self.alphas.append(policy.alpha.item())

        # <torch, + >
        self.policy_loss = np.append(self.policy_loss, policy_loss.detach().cpu().numpy())

        # <np, 1>
        return policy_loss.detach().cpu().numpy()

    # policy <NN>, value <NN>, dynamic <Dynamic>
    def value_update(self, policy, value, dynamic):
        # for i = 0 : 19
        for i in range(self.args.forward_step):
            logging.debug(i)
            if i == 0:
                # self.x_forward[i] = self.agent_batch.detach()
                # <torch, 256 * 6>
                self.x_forward[i] = self.state_batch.detach()
                logging.debug(self.x_forward[i])

            self.u_forward[i],self.longitudinal_v_forward[i] = policy.select_action(self.x_forward[i][:, 0:4])

            # <torch, 256 * 6>, <256 * 1>
            self.x_forward[i + 1], _, self.l_forward[i] = dynamic.step(self.x_forward[i], self.u_forward[i], self.longitudinal_v_forward[i],need_utility=True)
            # print('l_forward = ', self.l_forward[i].mean().item())

        # todo: why is state_batch?
        # <torch, 256 * 6>
        self.state_batch_next = self.x_forward[-1][:, 0:4]
        logging.debug(self.state_batch_next)

        # <torch, 256 * 1>
        self.value_next = value(self.state_batch_next)
        logging.debug(self.value_next)
        # print('value_next = ', self.value_next)

        # [0] <torch, 20 * 256>
        self.utility = torch.zeros([self.args.forward_step, self.args.batch_size], dtype=torch.float32).to(self.device)  # 初始化utility

        # for i = 0 : 19
        for i in range(self.args.forward_step):
            logging.debug(i)
            # <torch, 256>
            self.utility[i] = self.l_forward[i].clone()  # utility赋值
            logging.debug(self.l_forward[i])
            logging.debug(self.utility[i])

        # todo: 这里必须保持grad_dn = True
        # <torch, 256 * 1>
        self.sum_utility = torch.sum(self.utility, 0).unsqueeze(1)
        logging.debug(self.sum_utility)
        # print('sum_utility = ', self.sum_utility.mean().item())
        # <torch, 256 * 1>
        target_value = self.sum_utility.detach() + self.value_next.detach()
        logging.debug(target_value)

        # Input is state_batch
        self.state_batch.requires_grad_(False)

        # <torch, 256 * 1>
        self.current_value = value(self.state_batch[:, 0:4])
        logging.debug(self.current_value)

        # todo:如果增加了纵向控制，那么其平衡状态的速度不再是0！
        # [0, 0, 0, 0] <torch, 1 * 4>
        equilibrium_state = value._zero_state

        # <torch, 1>
        value_equilibrium = value(equilibrium_state)

        # <torch, 1> # J_critic
        value_loss = 1 / 2 * torch.mean(torch.pow((target_value - self.current_value), 2)) \
                     + 10 * torch.pow(value_equilibrium, 2)

        # for i = 0 : 1, 2 steps PEV #执行第二次时出错
        for i in range(1):
            value.zero_grad()
            value_loss.backward(retain_graph=True)  # false
            torch.nn.utils.clip_grad_norm_(value.parameters(), 3.0)
            value.optim.step()

        # <torch, + >
        self.value_loss = np.append(self.value_loss, value_loss.detach().cpu().numpy())

        # <np, 1 >
        return value_loss.detach().cpu().numpy()

    def plot_figure(self):

        print('len(self.value_loss)',len(self.value_loss))
        print('len(self.policy_loss)',len(self.policy_loss))
        print('len(self.alpha)',len(self.alphas))
        plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        save_dir = os.path.join('./parameters',self.args.method_version)
        os.makedirs(save_dir,exist_ok=True)

        plt.figure()
        plt.plot(range(len(self.value_loss)), self.value_loss / self.args.batch_size, label='value_loss')
        plt.xlabel('迭代次数')
        plt.ylabel('$L_V$')
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig(os.path.join(save_dir,'value_loss.png'),dpi=300)
        plt.close()

        plt.figure()
        plt.plot(range(len(self.policy_loss)), self.policy_loss / self.args.batch_size, label='policy_loss')
        plt.xlabel('迭代次数')
        plt.ylabel('$L_\pi$')
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig(os.path.join(save_dir,'policy_loss.png'),dpi=300)
        plt.close()

        plt.figure()
        plt.plot(range(len(self.alphas)), self.alphas, label='alpha')
        plt.xlabel('迭代次数')
        plt.ylabel(r'$\alpha$')
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig(os.path.join(save_dir,'alphas.png'),dpi=300)
        plt.close()

        args_dict = vars(self.args)
        with open(os.path.join(save_dir,self.args.method_version)+'.txt',"w") as f:
            for key, value in args_dict.items():
                f.write(f"{key}:{value}\n")