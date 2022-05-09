import stable_baselines3
import numpy as np
import os.path
import os
import copy
import sys

#models_dir = "models"

############################################################
#                    Get block types                       #
############################################################
char_to_block_type_info = {
    '+': ('positive', 1),
    '-': ('negative', -5),
    '0': ('neutral', 0)
}
block_types_string = sys.argv[1]
env_name_simple = sys.argv[2]
models_dir = sys.argv[3]
assert(len(block_types_string) == 3)
assert(all([x in char_to_block_type_info.keys() for x in block_types_string]))
block_types_strings = [str(char_to_block_type_info[x][0]) for x in block_types_string]
block_types_numbers = [str(char_to_block_type_info[x][1]) for x in block_types_string]
print('Block types:', ", ".join(block_types_strings))

############################################################
#                    Make environment                      #
############################################################
if env_name_simple == 'gridworld':
    from new_block_env import *
    env = NewBlockEnv(block_types)
    char_to_block_type_info = {
        '+': ('positive', 1, BlockType.POSITIVE),
        '-': ('negative', -5, BlockType.NEGATIVE),
        '0': ('neutral', 0, BlockType.NEUTRAL)
    }
    block_types = [char_to_block_type_info[x][2] for x in block_types_string]
elif env_name_simple == 'spaceinvaders':
    from spaceinvaders_block_env import *
    env = SpaceInvadersBlockEnv(block_types)
    char_to_block_type_info = {
        '+': ('positive', 1, BlockType.POSITIVE),
        '-': ('negative', -5, BlockType.NEGATIVE),
        '0': ('neutral', 0, BlockType.NEUTRAL)
    }
    block_types = [char_to_block_type_info[x][2] for x in block_types_string]
elif env_name_simple == 'miner':
    import procgen
    env = gym.make('procgen-miner-v0', extra_info = ','.join([str(x) for x in block_types_numbers]))
elif env_name_simple == 'dodgeball':
    import procgen
    env = gym.make('procgen-dodgeball-v0', extra_info = ','.join([str(x) for x in block_types_numbers]))

############################################################
#                    Load model                            #
############################################################
model_name_string = "_".join(block_types_strings) + ".zip"
model_path = models_dir + os.path.sep + model_name_string
print('Model path:', model_path)
assert(os.path.exists(model_path))
model = stable_baselines3.PPO.load(model_path, env)

############################################################
#                 Feature vector computations              #
############################################################
# WARNING modifies environment
def get_random_feature_vector_and_return(policy, environment, discount_factor = 1):
    feature_counts = np.zeros_like(environment._get_feature_counts())
    obs = env.soft_reset()
    current_discount = 1
    done = False
    ret = 0
    while not done:
        action, _ = policy.predict(obs)
        obs, reward, done, info = environment.step(action)
        feature_counts += info['features'] * current_discount
        ret += reward * current_discount
        current_discount *= discount_factor
    return {'feature_counts': feature_counts, 'return': ret}

############################################################
#                 Get all the half planes                  #
############################################################

def get_random_plane(policy, environment, **kwargs):
    rollouts = sorted([get_random_feature_vector_and_return(policy, environment) for _ in range(2)], key=lambda x: x['return'], reverse=True)
    if (rollouts[0]['feature_counts'] == rollouts[1]['feature_counts']).all() or rollouts[0]['return'] == rollouts[1]['return']:
        return get_random_plane(policy, environment, **kwargs) # Try again if we accidentally got the same thing
    else:
        return rollouts[0]['feature_counts'] - rollouts[1]['feature_counts']

if not os.path.exists('planedata_fulltraj_' + env_name_simple):
    os.mkdir('planedata_fulltraj_' + env_name_simple)

plane_normal_vectors = []
num_planes = 100 # arbitrary number
for i in range(num_planes):
    print('\r', i, '/', num_planes, '|', env.get_reward_weights(), end='')
    plane = get_random_plane(model, env)
    plane_normal_vectors.append(plane)
    with open("planedata_fulltraj_" + env_name_simple + "/normal_vectors_" + "_".join(block_types_strings) + ".txt", "w") as f:
        f.write(" ".join([str(x) for x in env.get_reward_weights()]) + "\n")
        f.write("\n".join([" ".join([str(x) for x in normal_vector]) for normal_vector in plane_normal_vectors]))

print("\rDone")
