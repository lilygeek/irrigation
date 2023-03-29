import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        
        self.max_action = max_action
        self.phi = phi


    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * torch.sigmoid(self.l3(a))
        return (a + action).clamp(0, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)


    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device


    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(state, z)

        return u, mean, std


    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)
        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.sigmoid(self.d3(a))
        


class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
        latent_dim = action_dim * 2

        self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device


    def select_action(self, state):        
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
            print(self.vae.decode(state))
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(0)
        return action[ind].cpu().data.numpy().flatten(),q1[ind].cpu().data.numpy().flatten(),action.cpu().data.numpy().flatten()


    def train(self, replay_buffer, iterations, batch_size=100):
        vae_losses = 0
        critic_losses = 0
        actor_losses = 0

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, action)
            #print(recon.size(),mean.size(),std.size())
            weight_0 = batch_size/2/torch.sum(action == 0)
            if torch.sum(action!=0)>0:
                weight_1 = batch_size/2/torch.sum(action != 0)
            else:
                weight_1 = 0
                weight_0 = 0
            #print(weight_0,weight_1) 
            ind_0 = (action ==0).reshape(-1)
            ind_1 = (action != 0).reshape(-1)
            action_0 = action[action==0]
            recon_0 = recon[action == 0]
            mean_0 = mean[ind_0,:]
            std_0 = std[ind_0,:]
            action_1 = action[action>0]
            recon_1 = recon[action> 0]
            mean_1 = mean[ind_1,:]
            std_1 = std[ind_1,:]           
            recon_loss_0 = F.mse_loss(recon_0, action_0)
            print(action.reshape(-1))
            KL_loss_0    = -0.5 * (1 + torch.log(std_0.pow(2)) - mean_0.pow(2) - std_0.pow(2)).mean()
            vae_loss_0 = recon_loss_0 + 0.5 * KL_loss_0
            
            recon_loss_1 = F.mse_loss(recon_1, action_1)
            KL_loss_1    = -0.5 * (1 + torch.log(std_1.pow(2)) - mean_1.pow(2) - std_1.pow(2)).mean()
            vae_loss_1 = recon_loss_1 + 0.5 * KL_loss_1

            vae_loss = weight_0*vae_loss_0+weight_1*vae_loss_1
            #print('loss',torch.sum(ind_0),vae_loss,recon_loss_0,recon_loss_1,KL_loss_0,KL_loss_1,'\n') 

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            vae_losses += vae_loss.item()


            # Critic Training
            with torch.no_grad():
                # Duplicate next state 10 times
                next_state = torch.repeat_interleave(next_state, 10, 0)

                # Compute value of perturbed actions sampled from the VAE
                target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

                # Soft Clipped Double Q-learning 
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
                # Take max over each action sampled from the VAE
                target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

                target_Q = reward + not_done * self.discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            critic_losses += critic_loss.item()




            # Pertubation Model / Action Training
            sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)

            # Update through DPG
            actor_loss = -self.critic.q1(state, perturbed_actions).mean()
              
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses += actor_loss.item()


            # Update Target Networks 
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return vae_losses/iterations, critic_losses/iterations, actor_losses/iterations 

    def test(self, state, action):
         
        selected_action,value,actions=self.select_action(state)
        state = np.reshape(state,[1,-1])
        action = np.reshape(action,[1,-1])
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        with torch.no_grad():
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss    = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            # Pertubation Model / Action Training
            return selected_action,value,vae_loss.item(),actions


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.vae.state_dict(), filename + "_vae")
        torch.save(self.vae_optimizer.state_dict(), filename + "_vae_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

        self.vae.load_state_dict(torch.load(filename + "_vae"))
        self.vae_optimizer.load_state_dict(torch.load(filename + "_vae_optimizer"))
        self.vae_target = copy.deepcopy(self.vae)
        
