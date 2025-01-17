import numpy as np
import random
import torch
import sys
import os
import argparse
import pathlib
import copy
from utils import get_config
from environment_syk_finite import CircuitEnv
import agents
import time
torch.set_num_threads(1)
import json

class Saver:
    def __init__(self, results_path, experiment_seed):
        self.stats_file = {'train': {}, 'test': {}}
        self.exp_seed = experiment_seed
        self.rpath = results_path

    def get_new_episode(self, mode, episode_no):
        if mode == 'train':
            self.stats_file[mode][episode_no] = {'loss': [],
                                                 'actions': [],
                                                 'errors': [],
                                                 'errors_noiseless':[],
                                                 'free_en_state':[],
                                                 'done_threshold': 0,
                                                 'bond_distance': 0,
                                                 'nfev': [], 
                                                 'expval' : [],
                                                 'entropy' : [],
                                                 'opt_ang': [],
                                                 'save_circ' : [],
                                                 'time' : [],
                                                 'state_fidelity': [],
                                                 }
        elif mode == 'test':
            self.stats_file[mode][episode_no] = {'actions': [],
                                                 'errors': [],
                                                 'errors_noiseless':[],
                                                 'done_threshold': 0,
                                                 'bond_distance': 0,
                                                 'nfev': [],
                                                 'opt_ang': [],
                                                 'time' : []
                                                 }

    def save_file(self):
        # np.save(f'{self.rpath}/summary_{self.exp_seed}.npy', self.stats_file)
        with open(f"{self.rpath}/summary_{self.exp_seed}.json", "w") as outfile:
            json.dump(self.stats_file, outfile)
        # with open('test_data_for_testing/sample.json', 'r') as openfile:
            # json_object = json.load(openfile)

    def validate_stats(self, episode, mode):
        assert len(self.stats_file[mode][episode]['actions']) == len(self.stats_file[mode][episode]['errors'])

    
def modify_state(state,env):
    # print(state.shape)
        
    if conf['agent']['en_state']:
        batch= torch.tensor([float(env.prev_energy)]*state.shape[1], dtype=torch.float, device=device).view(state.shape[1])[None, :, None]
        state = torch.cat((state, batch.repeat(state.shape[0], 1, 1)), dim=2)
        state = state.unsqueeze(0)

    if "threshold_in_state" in conf['agent'].keys() and conf['agent']["threshold_in_state"]:
        batch= torch.tensor([float(env.done_threshold)]*state.shape[1], dtype=torch.float, device=device).view(state.shape[1])[None, :, None]
        state = torch.cat((state, batch.repeat(state.shape[0], 1, 1)), dim=2)
        state = state.unsqueeze(0)
    
    # print(state.shape)
    # exit()
    return state


def agent_test(env, agent, episode_no, seed, output_path,threshold):
    """ Testing function of the trained agent. """    
    agent.saver.get_new_episode('test', episode_no)
    state = env.reset()
    state = modify_state(state, env)
    current_epsilon = copy.copy(agent.epsilon)
    agent.policy_net.eval()

    for t in range(env.num_layers + 1):
        ill_action_from_env = env.illegal_action_new()
        
        agent.epsilon = 0
        with torch.no_grad():
            action, _ = agent.act(state, ill_action_from_env)
            assert type(action) == int
            agent.saver.stats_file['test'][episode_no]['actions'].append(action)
        next_state, reward, done = env.step(agent.translate[action],train_flag=False)
        next_state = modify_state(next_state, env)
        state = next_state.clone()
        assert type(env.error) == float 
        agent.saver.stats_file['test'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['test'][episode_no]['errors_noiseless'].append(env.error_noiseless)
        agent.saver.stats_file['test'][episode_no]['opt_ang'].append(env.opt_ang_save)
        
        if done:
            
            agent.saver.stats_file['test'][episode_no]['done_threshold'] = env.done_threshold
            # agent.saver.stats_file['test'][episode_no]['bond_distance'] = env.current_bond_distance
            errors_current_bond = [val['errors'][-1] for val in agent.saver.stats_file['test'].values()
                                   if val['done_threshold'] == env.done_threshold]
            if len(errors_current_bond) > 0 and min(errors_current_bond) > env.error:
                torch.save(agent.policy_net.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_best_geo_{env.current_prob}_model.pth")
                torch.save(agent.optim.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_best_geo_{env.current_prob}_optim.pth")
            agent.epsilon = current_epsilon
            agent.saver.validate_stats(episode_no, 'test')
            
            return reward, t
        

def one_episode(episode_no, env, agent, episodes):
    """ Function preforming full trainig episode."""
    t0 = time.time()
    agent.saver.get_new_episode('train', episode_no)
    state = env.reset()
    # agent.saver.stats_file['train'][episode_no]['bond_distance'] = env.current_bond_distance
    agent.saver.stats_file['train'][episode_no]['done_threshold'] = env.done_threshold
    
    state = modify_state(state, env)
    agent.policy_net.train()
    rewards4return = []
    
    for itr in range(env.num_layers + 1):
        ill_action_from_env = env.illegal_action_new()
        
        action, _ = agent.act(state, ill_action_from_env)
        assert type(action) == int
        agent.saver.stats_file['train'][episode_no]['actions'].append(action)
        
        next_state, reward, done = env.step(agent.translate[action])
        
        next_state = modify_state(next_state, env)
        agent.remember(state, 
                       torch.tensor(action, device=device), 
                       reward,
                       next_state,
                       torch.tensor(done, device=device))
        state = next_state.clone()
        rewards4return.append(float(reward.clone()))

        assert type(env.error) == float
        agent.saver.stats_file['train'][episode_no]['errors'].append(env.error)
        agent.saver.stats_file['train'][episode_no]['errors_noiseless'].append(env.error_noiseless)
        agent.saver.stats_file['train'][episode_no]['expval'].append(env.expval)
        agent.saver.stats_file['train'][episode_no]['entropy'].append(env.entropy)
        agent.saver.stats_file['train'][episode_no]['save_circ'].append(env.save_circ)
        agent.saver.stats_file['train'][episode_no]['free_en_state'].append(env.free_en_state)
        agent.saver.stats_file['train'][episode_no]['time'].append(time.time()-t0)
        agent.saver.stats_file['train'][episode_no]['state_fidelity'].append(env.fidelity)

        # wandb.log(
        # {"train_by_step/step_no": itr,
        # "train_by_step/episode_no": episode_no,
        # "train_by_step/errors": env.error,
        # "train_by_step/errors_noiseless": env.error_noiseless,
        # })

        if agent.memory_reset_switch:            
           if env.error < agent.memory_reset_threshold:
               agent.memory_reset_counter += 1
           if agent.memory_reset_counter == agent.memory_reset_switch:
               agent.memory.clean_memory()
               agent.memory_reset_switch = False
               agent.memory_reset_counter = False
               
  
        if done:

            # wandb.log(
            #     {"train_final/episode_len": itr,
            #     "train_final/errors": env.error,
            #     "train_final/errors_noiseless": env.error_noiseless,
            #     "train_final/done_threshold": env.done_threshold,
            #     "train_final/bond_distance": env.current_bond_distance,
            #     "train_final/episode_no": episode_no,
            #     "train_final/current_number_of_cnots": env.current_number_of_cnots,
            #     "train_final/epsilon": agent.epsilon,
            #     "train_final/return": sum([x*y for x,y in zip(rewards4return,[agent.gamma**i for i in range(1,len(rewards4return)+1)])]),
            #     "train_final/rwd_sum": sum(rewards4return),

            #     })
            # print('time:', time.time()-t0)
            if episode_no%5==0:
                print("episode: {}/{}, score: {}, e: {:.2}, rwd: {} \n"
                        .format(episode_no, episodes, itr, agent.epsilon, reward),flush=True)
            break 
        
        if len(agent.memory) > conf['agent']['batch_size']:
            if "replay_ratio" in conf['agent'].keys():
                if  itr % conf['agent']["replay_ratio"]==0:
                    loss = agent.replay(conf['agent']['batch_size'])
            else:
                loss = agent.replay(conf['agent']['batch_size'])         
            assert type(loss) == float
            agent.saver.stats_file['train'][episode_no]['loss'].append(loss)
            agent.saver.validate_stats(episode_no, 'train')
            # wandb.log({"train_by_step/loss":loss})
            
            

def train(agent, env, episodes, seed, output_path,threshold):
    """Training loop"""
    threshold_crossed = 0
    for e in range(episodes):
        
        one_episode(e, env, agent, episodes)
        
        if e %20==0 and e > 0:
            agent.saver.save_file()
            torch.save(agent.policy_net.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_model.pth")
            torch.save(agent.optim.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_optim.pth")
            torch.save( {i: a._asdict() for i,a in enumerate(agent.memory.memory)}, f"{output_path}/thresh_{threshold}_{seed}_replay_buffer.pth")
        if env.error <= 0.0016:
            threshold_crossed += 1
            np.save( f'threshold_crossed', threshold_crossed )

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproduction')
    parser.add_argument('--config', type=str, default='h_s_2', help='Name of configuration file')
    parser.add_argument('--experiment_name', type=str, default='lower_bound_energy/', help='Name of experiment')
    parser.add_argument('--gpu_id', type=int, default=0, help='Set specific GPU to run experiment [0, 1, ...]')
    # parser.add_argument('--wandb_group', type=str, default='test/', help='Group of experiment run for wandb')
    # parser.add_argument('--wandb_name', type=str, default='test/', help='Name of experiment run for wandb')
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    conf = get_config(args.experiment_name, f'{args.config}.cfg')
    if conf['agent']['init_net']: 
        results_path =f"results/{args.experiment_name}{args.config}/"
    else:
        results_path =f"results/"
    pathlib.Path(f"{results_path}{args.experiment_name}{args.config}").mkdir(parents=True, exist_ok=True)
    # device = torch.device(f"cuda:{args.gpu_id}")
    device = torch.device(f"cpu:0")
    
    
   

    loss_dict, scores_dict, test_scores_dict, actions_dict = dict(), dict(), dict(), dict()
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # wandb_project = 
    # wandb_entity = 

    # wandb.login()
    # run = wandb.init(project=wandb_project,
    #                 config=conf,
    #                 entity= wandb_entity,
    #                 group=args.wandb_group,
    #                 name=args.wandb_name)
    

    actions_test = []
    action_test_dict = dict()
    error_test_dict = dict()
    error_noiseless_test_dict=dict()

    
    """ Environment and Agent initialization"""
    environment = CircuitEnv(conf, device=device)
    agent = agents.__dict__[conf['agent']['agent_type']].__dict__[conf['agent']['agent_class']](conf, environment.action_size, environment.state_size, device)
    agent.saver = Saver(f"{results_path}{args.experiment_name}{args.config}", args.seed)
    # print(results_path)
    if conf['agent']['init_net']: 
        PATH = f"{results_path}thresh_2.9_{args.seed}"
        # print(PATH)
        # exit()
        agent.policy_net.load_state_dict(torch.load(PATH+f"_model.pth"))
        agent.target_net.load_state_dict(torch.load(PATH+f"_model.pth"))
        agent.optim.load_state_dict(torch.load(PATH+f"_optim.pth"))
        agent.policy_net.eval()
        agent.target_net.eval()

        replay_buffer_load = torch.load(f"{PATH}_replay_buffer.pth")
        for i in replay_buffer_load.keys():
            agent.remember(**replay_buffer_load[i])

        if not conf['agent']['epsilon_restart']:
            agent.epsilon = agent.epsilon_min

    train(agent, environment, conf['general']['episodes'], args.seed, f"{results_path}{args.experiment_name}{args.config}",conf['env']['accept_err'])
    agent.saver.save_file()
            
    torch.save(agent.policy_net.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_model.pth")
    torch.save(agent.optim.state_dict(), f"{results_path}{args.experiment_name}{args.config}/thresh_{conf['env']['accept_err']}_{args.seed}_optim.pth")

    # wandb.finish()