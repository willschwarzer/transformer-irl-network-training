import os
import sys
import time
from utils import get_freest_gpu, convert_chai_rollouts
FREEST_GPU = get_freest_gpu()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FREEST_GPU)
import argparse
import numpy as np
from predict_transformer_nonlinear import NonLinearNet
import stable_baselines3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.types import load
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import torch
import torch.nn.functional as F
from new_block_env_new import NewBlockEnv
import ray
import wandb

NUM_AGENTS_PER_GPU = 10
NUM_ROLLOUTS_PER_AGENT = 10000

def parse_args():
    parser = argparse.ArgumentParser(description='Train supervised IRL models')
    parser.add_argument('--data', type=str, default='grid_regression_with_weights_chai_100000',
                        help='location of rollouts')
    parser.add_argument('--weights', type=str, default='weights_chai_100000.npy')
    parser.add_argument('--model', '-sm', type=str, default='ground_truth_phi_test.parameters', help='location of saved model to load')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--wandb-project', '-wp', type=str, default='sirl')
    parser.add_argument('--max-threads', '-mt', default=50, type=int)
    parser.add_argument('--average-traj-reps', '-atr', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--env', default='rings', type=str)
    args = parser.parse_args()
    return args

@ray.remote(num_gpus=1/(NUM_AGENTS_PER_GPU))
def evaluate_random(env):
    model = stable_baselines3.PPO("MlpPolicy", env, verbose=1)
    mean_random_ret, _ = evaluate_policy(model, env, n_eval_episodes=100)
    return mean_random_ret

@ray.remote(num_gpus=1/(NUM_AGENTS_PER_GPU))
def evaluate_random_rew(env, model_name):
    print("USING models_regression BY DEFAULT")
    model = stable_baselines3.PPO.load("models_regression" + "/" + model_name, env)
    mean_random_ret, _ = evaluate_policy(model, env, n_eval_episodes=100)
    return mean_random_ret

@ray.remote(num_gpus=1/(NUM_AGENTS_PER_GPU))
def evaluate_ground_truth(env):
    agent = stable_baselines3.PPO("MlpPolicy", env, verbose=1)
    agent.learn(total_timesteps=100000)
    mean_ground_truth_ret, _ = evaluate_policy(agent, env, n_eval_episodes=100)
    return mean_ground_truth_ret

@ray.remote(num_gpus=1/(NUM_AGENTS_PER_GPU))
def evaluate_sirl(traj_rep, state_encoder, env):
    pred_env = NewBlockEnv(traj_rep, state_encoder)
    pred_model = stable_baselines3.PPO("MlpPolicy", pred_env, verbose=1)
    pred_model.learn(total_timesteps=100000)
    mean_ground_truth_pred_model_ret, _ = evaluate_policy(pred_model, env, n_eval_episodes=100)
    return mean_ground_truth_pred_model_ret

@ray.remote(num_gpus=1/(NUM_AGENTS_PER_GPU))
def evaluate_adversarial(rollouts, env, alg):
    trainer = GAIL if alg == 'gail' else AIRL
    
    venv = DummyVecEnv([lambda: env] * 8)
    
    learner = stable_baselines3.PPO(
        env=venv,
        policy='MlpPolicy',
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
    )
    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )
    adversarial_trainer = trainer(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )
    adversarial_trainer.train(1000000)
    mean_ret, _ = evaluate_policy(learner, env, n_eval_episodes=1)
    return mean_ret
    

def load_model(location, obs_size, horizon):
    trajectory_rep_dim = 100
    state_rep_dim = 100
    state_hidden_size = 2048
    num_state_layers = 2
    trajectory_hidden_size = 2048
    num_trajectory_layers = 2
    mlp = False
    trajectory_sigmoid = False
    lstm = False
    repeat_trajectory_calculations = False
    ground_truth_phi = False
    
    net = NonLinearNet(trajectory_rep_dim, state_rep_dim, state_hidden_size, 64, trajectory_hidden_size, obs_size, horizon, num_trajectory_layers, num_state_layers, mlp=mlp, trajectory_sigmoid=trajectory_sigmoid, lstm=lstm, repeat_trajectory_calculations=repeat_trajectory_calculations, ground_truth_phi = ground_truth_phi).cuda() # Maybe shouldn't make this cuda
    net.load_state_dict(torch.load(location))
    return net

def main(args):
    wandb.init(project=args.wandb_project)
    if "grid" in args.env:
        obs_size = 4*25
        horizon = 150
    elif "space" in args.env:
        obs_size = 6*25
        horizon = 150
    elif "ring" in args.env or "object" in env:
        obs_size = 2*args.num_rings
        horizon = 50
    else:
        raise NotImplementedError
    model = load_model(args.model, obs_size, horizon)
    state_encoder = model.state_encoder
    trajectory_encoder = model.trajectory_encoder
    weights = np.load(f"data/{args.weights}")
    # weight_batches_redundant = weights.reshape(-1, 100, 3)
    # weight_batches = weight_batches_redundant[:, 0, :].squeeze()
    print(weights.shape)
    rollouts = load(f"data/{args.data}")
    states, _ = convert_chai_rollouts(rollouts)
    rollout_batches = [rollouts[i:(i+NUM_ROLLOUTS_PER_AGENT)] for i in range(len(rollouts)//NUM_ROLLOUTS_PER_AGENT)]
    state_batches_np = states.reshape(-1, NUM_ROLLOUTS_PER_AGENT, 150, 25)
    state_batches_int = torch.Tensor(state_batches_np).to(torch.int64)
    state_batches = F.one_hot(state_batches_int, num_classes=4).view(-1, NUM_ROLLOUTS_PER_AGENT, 150, 100).to(torch.float)
    models_directory = "models_regression"
    models = os.listdir(models_directory)
    np.random.shuffle(models)
    models = [models[i:(i+10)] for i in range(len(models)//10)]
    ray.init(num_cpus=10, num_gpus=1)
    for rollout_batch, traj_batch, weights, model_batch in zip(rollout_batches, state_batches, weights, models):
        ground_truth_env = NewBlockEnv(weights)
        # Random agent
        # random_rets = ray.get([evaluate_random.remote(ground_truth_env) for i in range(100)])
        # ave_random_ret = np.mean(random_rets)
        
        # Trained agent on random rewards
        random_rew_rets = ray.get([evaluate_random_rew.remote(ground_truth_env, model_name) for model_name in model_batch])
        ave_random_rew_ret = np.mean(random_rew_rets)
        
        # Ground truth
        ground_truth_rets = ray.get([evaluate_ground_truth.remote(ground_truth_env) for i in range(10)])
        ave_ground_truth_ret = np.mean(ground_truth_rets)
        
        # Our method
        if args.average_traj_reps:
            traj_rep = trajectory_encoder(traj_batch[:256].cuda()).mean(axis=0).cpu().detach().numpy()
        else:
            traj_rep = trajectory_encoder(traj_batch[0].unsqueeze(0).cuda()).squeeze().cpu().detach().numpy()
        sirl_rets = ray.get([evaluate_sirl.remote(traj_rep, state_encoder, ground_truth_env) for i in range(10)])
        ave_sirl_ret = np.mean(sirl_rets)
        
        # GAIL
        gail_rets = ray.get([evaluate_adversarial.remote(rollout_batch, ground_truth_env, 'gail') for i in range(10)])
        ave_gail_ret = np.mean(gail_rets)
        
        # AIRL
        airl_rets = ray.get([evaluate_adversarial.remote(rollout_batch, ground_truth_env, 'airl') for i in range(10)])
        ave_airl_ret = np.mean(airl_rets)
        
        # wandb.log({'random_ret': ave_random_ret, 'ground_truth_ret': ave_ground_truth_ret, 'sirl_ret': ave_sirl_ret, 'gail_ret': ave_gail_ret, 'airl_ret': ave_airl_ret})
        wandb.log({'random_rew_ret': ave_random_rew_ret, 'ground_truth_ret': ave_ground_truth_ret, 'sirl_ret': ave_sirl_ret, 'gail_ret': ave_gail_ret, 'airl_ret': ave_airl_ret})

if __name__ == "__main__":
    args = parse_args()
    main(args)