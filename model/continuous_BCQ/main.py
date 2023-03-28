import argparse
import gym
import numpy as np
import time
import os
import torch
import pandas as pd
import BCQ
import DDPG
import utils
from torch.utils.tensorboard import SummaryWriter


# Handles interactions with the environment, i.e. train behavioral or generate buffer
def interact_with_environment(env, state_dim, action_dim, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize and load policy
    policy = DDPG.DDPG(state_dim, action_dim, max_action, device)#, args.discount, args.tau)
    if args.generate_buffer: policy.load(f"./models/behavioral_{setting}")

    # Initialize buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    
    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Interact with the environment for max_timesteps
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action with noise
        if (
            (args.generate_buffer and np.random.uniform(0, 1) < args.rand_action_p) or 
            (args.train_behavioral and t < args.start_timesteps)
        ):
            action = env.action_space.sample()
        else: 
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.gaussian_std, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if args.train_behavioral and t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if args.train_behavioral and (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/behavioral_{setting}", evaluations)
            policy.save(f"./models/behavioral_{setting}")

    # Save final policy
    if args.train_behavioral:
        policy.save(f"./models/behavioral_{setting}")

    # Save final buffer and performance
    else:
        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/buffer_performance_{setting}", evaluations)
        replay_buffer.save(f"./buffers/{buffer_name}")


# Trains BCQ offline
def train_BCQ(state_dim, action_dim, max_action, device, args):
    # For saving files
    setting = f"{args.env}_{args.seed}"
    buffer_name = f"{args.buffer_name}_{setting}"

    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(f"./buffers/{buffer_name}")
    
    evaluations = []
    episode_num = 0
    done = True 
    training_iters = 0
    
    while training_iters < args.max_timesteps: 
        pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq), batch_size=args.batch_size)

        evaluations.append(eval_policy(policy, args.env, args.seed))
        np.save(f"./results/BCQ_{setting}", evaluations)

        training_iters += args.eval_freq
        print(f"Training iterations: {training_iters}")

# Trains BCQ offline
def train_BCQ_offline(device, args):

    # Load buffer
    #replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    #replay_buffer.load(f"./buffers/{buffer_name}")

    params = [1,1,1,1]

    data = pd.read_excel('../../data/main_hourly_locomos.xlsx')


    envs = data[['moisture9','moisture18','moisture24','soilwater','soiltemp','lws','temp','humid','pure_prep']]

    irrigation = data['irrigation']

    envs = envs.to_numpy()[0:-24,:]
    irrigation =np.absolute(irrigation.to_numpy().reshape(-1,1))[0:-24]
    rewards = np.zeros(envs.shape[0]-25)
    m_r1 = np.abs(envs[:,3]/2.6-1)
    m_r2 = m_r1[1:]
    m_r1 = m_r1[0:-1]

    m9_v1 = envs[0:-1,0]
    m9_v2  = envs[1:,0]
    m12_v1 = envs[0:-1,1]
    m12_v2  = envs[1:,1]
    m24_v1 = envs[0:-1,2]
    m24_v2  = envs[1:,2]
    water = envs[0:-1,3]
    desire_bool = np.zeros(envs.shape[0]-1)
    irrigation_bool = irrigation[0:-1] ==0
    irrigation_bool = irrigation_bool.reshape(-1)
    irrigation_water = irrigation[0:-1].reshape(-1)
    m9v_bool = np.zeros(envs.shape[0]-1)
    m12v_bool = np.zeros(envs.shape[0]-1)
    m24v_bool = np.zeros(envs.shape[0]-1)
    desire_bool[water<2.6] = 1
    m9v_bool[(m9_v2-m9_v1)>0] = 1
    m12v_bool[(m12_v2-m12_v1)>0] = 1
    m12v_bool[(m24_v2-m24_v1)!=0] = 1
    rewards = params[0]*desire_bool*(m_r1-m_r2)+(1-desire_bool)*irrigation_bool-params[1]*np.abs(m24_v2-m24_v1)+params[2]*desire_bool*(m9_v2-m9_v1)+params[3]*desire_bool*(m12_v2-m12_v1)+desire_bool*1/(irrigation_water+0.001)
    
    state_dim = envs.shape[1]
    action_dim = irrigation.shape[1]
    max_action = 1

    # Initialize buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    episode_reward = 0

    for i in range(envs.shape[0]-1):
        state = envs[i,:];
        action = irrigation[i];
        next_state = envs[i+1,:];
        reward = rewards[i];
        episode_reward += reward
        done_bool = i%24==23 or i==envs.shape[0]-2
        if done_bool:
            print(f"Episode reward:{episode_reward}")
            episode_reward = 0
        replay_buffer.add(state, action, next_state, reward, done_bool)

    replay_buffer.save(args.buffer_file+args.buffer_name+"_1111")


    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

    
    evaluations = []
    episode_num = 0
    done = True 
    training_iters = 0
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(args.log_file,'runs','BCQ'+'_'+timestr))
    path = args.model_file+"BCQ_"+timestr+"/"
    os.makedirs(path,exist_ok = True)
    while training_iters < args.max_timesteps: 
        vae, critic, actor = policy.train(replay_buffer, iterations=args.eval_freq, batch_size=args.batch_size)
          
        if training_iters%args.save_interval == 0:
            path_model = path+"model_"+str(round(training_iters/args.save_interval))
            policy.save(path_model)
        
        training_iters += args.eval_freq

        writer.add_scalar('VAE Loss',vae,training_iters)
        writer.add_scalar('Crtitic Loss',critic,training_iters)
        writer.add_scalar('Actor loss',actor,training_iters)
        print(f"Training iterations: {training_iters}")
        print("VAE Loss: ", vae, " Crtitic Loss: ", critic, " Actor loss: ",actor, "\n")


def test_BCQ_offline(device, args):

    # Load buffer
    #replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    #replay_buffer.load(f"./buffers/{buffer_name}")

    data = pd.read_excel('../../data/main_hourly_locomos.xlsx')


    envs = data[['moisture9','moisture18','moisture24','soilwater','soiltemp','lws','temp','humid','pure_prep']]

    irrigation = data['irrigation']

    envs = envs.to_numpy()[-72:-48,:]
    irrigation =np.absolute(irrigation.to_numpy().reshape(-1,1))[-24:]

    state_dim = envs.shape[1]
    action_dim = irrigation.shape[1]
    max_action = 1

    # Initialize policy
    policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

    policy.load(args.policy_file)

    
    for i in range(envs.shape[0]-1):
        state = envs[i,:];
        action = irrigation[i];
         
        action, value, vae_loss,actions = policy.test(state, action)
        print("Hour",i,"Water applied: ", action, " Estimated Q-value: ", value, " VAE Loss: ",vae_loss, "Candidate actions: ",actions,"\n")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", default="../../result/log/")
    parser.add_argument("--model_file", default="../../result/model/")
    parser.add_argument("--policy_file", default="../../result/model/")
    parser.add_argument("--buffer_file", default="../../result/buffer/")   
    parser.add_argument("--env", default="Hopper-v3")               # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_name", default="Main_Locomos")          # Prepends name to filename
    parser.add_argument("--eval_freq", default=100, type=int)     # How often (time steps) we evaluate
    parser.add_argument("--save_interval", default=500, type=int)     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=100000, type=int)   # Max time steps to run environment or train for (this defines buffer size)
    parser.add_argument("--start_timesteps", default=25000, type=int)# Time steps initial random policy is used before training behavioral
    parser.add_argument("--rand_action_p", default=0.3, type=float) # Probability of selecting random action during batch generation
    parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
    parser.add_argument("--batch_size", default=50, type=int)      # Mini batch size for networks
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--phi", default=0.05)                      # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
    parser.add_argument("--offline_data", action="store_true")  # If true, load offline_data
    parser.add_argument("--generate_buffer", action="store_true")   # If true, generate buffer
    parser.add_argument("--test_offline", action="store_true") 
    args = parser.parse_args()

    print("---------------------------------------")    
    if args.train_behavioral:
        print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
    elif args.generate_buffer:
        print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
    elif args.offline_data:
        print(f"Setting: Loading offline data into replay buffer")
    else:
        print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.train_behavioral and args.generate_buffer:
        print("Train_behavioral and generate_buffer cannot both be true.")
        exit()


    #env = gym.make(args.env)

    #env.seed(args.seed)
    #env.action_space.seed(args.seed)
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train_behavioral or args.generate_buffer:
        interact_with_environment(env, state_dim, action_dim, max_action, device, args)
    if args.offline_data:
        train_BCQ_offline(device, args)
    elif args.test_offline:
        test_BCQ_offline(device, args)
    else:
        train_BCQ(state_dim, action_dim, max_action, device, args)
