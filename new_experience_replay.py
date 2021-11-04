import argparse
import os

import gym
import time

from a2c.model import ActorCritic
from a2c.monitor import Monitor
from a2c.multiprocessing_env import SubprocVecEnv, VecPyTorch, VecPyTorchFrameStack
from a2c.wrappers import *
import torch
import torch.optim as optim

from atari_wrappers import make_atari, wrap_deepmind

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser("A2C experiments for Atari games")
    parser.add_argument("--seed", type=int, default=42, help="which seed to use")
    # Environment
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4", help="name of the game")
    # Core A2C parameters
    parser.add_argument("--actor-loss-coefficient", type=float, default=1.0, help="actor loss coefficient")
    parser.add_argument("--critic-loss-coefficient", type=float, default=0.5, help="critic loss coefficient")
    parser.add_argument("--entropy-loss-coefficient", type=float, default=0.01, help="entropy loss coefficient")
    parser.add_argument("--lr", type=float, default=7e-4, help="learning rate for the RMSprop optimizer")
    parser.add_argument("--alpha", type=float, default=0.99, help="alpha term the RMSprop optimizer")
    parser.add_argument("--eps", type=float, default=1e-5, help="eps term for the RMSprop optimizer")  # instead of 1e-3 due to different RMSprop implementation in PyTorch than Tensorflow
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="maximum norm of gradients")
    parser.add_argument("--num_steps", type=int, default=5, help="number of forward steps")
    parser.add_argument("--num-envs", type=int, default=6, help="number of processes for environments")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--num-frames", type=int, default=int(10e6),
                        help="total number of steps to run the environment for")
    parser.add_argument("--log-dir", type=str, default="logs", help="where to save log files")
    parser.add_argument("--save-freq", type=int, default=0, help="updates between saving models (default 0 => no save)")
    # Reporting
    parser.add_argument("--print-freq", type=int, default=1000, help="evaluation frequency.")
    return parser.parse_args()


def compute_returns(next_value, rewards, masks, gamma):
    # Critic value is a prediction of future rewards, 
    # so each r = immediate reward * discount factor * expected future rewards * mask (0/1 depending on if action resulted in done)
    r = next_value
    returns = []
    # calculated discounted returns in reverse order? why? can be removed without any clear difference
    for step in reversed(range(len(rewards))):
        #print(step)
        # For each workers reward, calculate the discounted reward (There is a reward in rewards for each worker [0,0,0,0] for 4 workers)
        r = rewards[step] + gamma * r * masks[step]
        # insert 0, inserts element at index 0
        returns.insert(0, r)
    return returns


def make_env(seed, rank):
    def _thunk():
        env = gym.make(args.env)
        #env.render(render_mode='human')
        #env = make_atari("BreakoutNoFrameskip-v4")
        # Warp the frames, grey scale, stake four frame and scale to smaller ratio
        #env = wrap_deepmind(env, frame_stack=True, scale=True)
        env.seed(seed + rank)
        assert "NoFrameskip" in args.env, "Require environment with no frameskip"
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)

        allow_early_resets = False
        log_dir = args.log_dir
        assert args.log_dir is not None, "Log directory required for Monitor! (which is required for episodic return reporting)"
        try:
            os.mkdir(log_dir)
        except:
            pass
        env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets)

        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        # env = PyTorchFrame(env)
        env = ClipRewardEnv(env)
        # env = FrameStack(env, 4)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env
    return _thunk


def make_envs():
    envs = [make_env(args.seed, i) for i in range(args.num_envs)]
    envs = SubprocVecEnv(envs)
    envs = VecPyTorch(envs, device)
    envs = VecPyTorchFrameStack(envs, 4, device)
    return envs

def deconvert():
    print("deconverted to gym")

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    envs = make_envs()

    actor_critic = ActorCritic(envs.observation_space, envs.action_space).to(device)
    optimizer = optim.RMSprop(actor_critic.parameters(), lr=args.lr, eps=args.eps, alpha=args.alpha)

    num_updates = args.num_frames // args.num_steps // args.num_envs # the // (integer division operator) divides and gives a integer value back 100 // 20 // 2 =2

    observation = envs.reset()
    start = time.time()

    # Doubly Ended Queue: 
    # Deque is preferred over list in the cases where we need quicker append and pop operations from both the ends of container, 
    # as deque provides an O(1) time complexity for append and pop
    # operations as compared to list which provides O(n) time complexity.
    episode_rewards = deque(maxlen=10)
    er_batches = []

    
    for update in range(num_updates):
        print(er_batches)
        # Initializes and resets data collected needed for each update
        log_probs = []
        values = []
        rewards = []
        actions = []
        masks = []
        entropies = []

        for step in range(args.num_steps):
            # if no convertion:
            #  RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation:
            #  [torch.cuda.FloatTensor [16, 4, 84, 84]] is at version 12; expected version 10 instead. Hint: enable anomaly
            #  detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
            observation = observation.to(device) / 255.
            actor, value = actor_critic(observation)

            action = actor.sample()
            # Gets next state, reward, done and info for each environment (returned as torch tensors with lists of the data inside) 
            # infos example (4 workers): ({'lives': 2}, {'lives': 4}, {'lives': 4}, {'lives': 3})
            next_observation, reward, done, infos = envs.step(action.unsqueeze(1))

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            #print("action") tensor([3, 0, 2, 2, 0, 1, 1, 3, 3, 3, 0, 3, 0, 2, 2, 1], device='cuda:0')
            log_prob = actor.log_prob(action)
            # print("actor") Categorical(probs: torch.Size([16, 4]), logits: torch.Size([16, 4]))            
            ''' print("log prob") 
                tensor([-1.3639, -1.3915, -1.3858, -1.3916, -1.3916, -1.4042, -1.3916, -1.3916,
                -1.4047, -1.4047, -1.3858, -1.3859, -1.3639, -1.3631, -1.3916, -1.4045],
                device='cuda:0', grad_fn=<SqueezeBackward1>)'''
            entropy = actor.entropy()
            #print(entropy)

            # Get 0 reward from action if it resulted in done (adding a mask where action function estimate will be multiplied by 0 or 1)
            mask = torch.from_numpy(1.0 - done).to(device).float()

            entropies.append(actor.entropy())
            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(reward.to(device).squeeze()) #torch.Size([16]) 3 timesteps with 4 workers [tensor([0., 0., 0., 0.], device='cuda:0'), tensor([0., 0., 0., 0.], device='cuda:0'), tensor([0., 0., 0., 0.], device='cuda:0')]
            masks.append(mask)

            observation = next_observation

        next_observation = next_observation.to(device).float() / 255.
        
        # Adds this updates worth of memories to 
        batch = {}
        batch['entropies'] = entropies
        batch['log_probs'] = log_probs
        batch['values'] = values
        batch['rewards'] = rewards
        batch['masks'] = masks
        batch['next_observation'] = next_observation
        er_batches.append(batch)


        # Pick random batch
        batch_len = len(er_batches)
        random_batch_index = np.random.choice(range(batch_len), size=1)[0]
        random_batch = er_batches[random_batch_index]
        # Compute discounted returns, uses a critic estimation as future rewards expected for that state
        with torch.no_grad():
            # Next values: Critic value / estimation of future rewards from this point onwards for this state
            _, next_values = actor_critic(random_batch['next_observation'])  
            returns = compute_returns(next_values.squeeze(), random_batch['rewards'], random_batch['masks'], args.gamma)
            returns = torch.cat(returns)

        # converts lists to torch tensor
        log_probs = torch.cat(random_batch['log_probs']) 
        values = torch.cat(random_batch['values'])
        entropies = torch.cat(random_batch['entropies'])

        # returns: calculated by actual returns + estimated returns * discount factor
        # values: the critic value alone calculated by the worker in that timestamp
        advantages = returns - values
        # define losses acording to A2C loss function
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = entropies.mean()

        loss = args.actor_loss_coefficient * actor_loss + \
               args.critic_loss_coefficient * critic_loss - \
               args.entropy_loss_coefficient * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
        optimizer.step()
        print("passed")
        '''
        if len(episode_rewards) > 1 and update % args.print_freq == 0:
            end = time.time()
            total_num_steps = (update + 1) * args.num_envs * args.num_steps
            print("********************************************************")
            print("update: {0}, total steps: {1}, FPS: {2}".format(update, total_num_steps, int(total_num_steps / (end - start))))
            print("mean/median reward: {:.1f}/{:.1f}".format(np.mean(episode_rewards), np.median(episode_rewards)))
            print("min/max reward: {:.1f}/{:.1f}".format(np.min(episode_rewards), np.max(episode_rewards)))
            print("actor loss: {:.5f}, critic loss: {:.5f}, entropy: {:.5f}".format(actor_loss.item(), critic_loss.item(), entropy_loss.item()))
            print("********************************************************")
        if args.save_freq > 0 and update % args.save_freq == 0:
            torch.save(actor_critic.state_dict(), 'models/{}-{}.pth'.format(args.env, update))
        '''