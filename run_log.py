# -*- coding:utf-8  -*-
import argparse
import os
import time
import json
import sys
import random
import numpy as np
import pickle as pkl
import torch as th
import multiprocessing
import tkinter as tk
from tkinter import simpledialog
import gfootball.env as football_env
from env.chooseenv import make
from utils.get_logger import get_logger
from env.obs_interfaces.observation import obs_type

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_players_and_action_space_list(g):
    if sum(g.agent_nums) != g.n_player:
        raise Exception("Agent number = %d is incorrect and does not match n_player = %d." % (sum(g.agent_nums), g.n_player))

    n_agent_num = list(g.agent_nums)
    for i in range(1, len(n_agent_num)):
        n_agent_num[i] += n_agent_num[i - 1]

    # Assign player IDs based on the agent number.
    players_id = []
    actions_space = []
    for policy_i in range(len(g.obs_type)):
        if policy_i == 0:
            players_id_list = range(n_agent_num[policy_i])
        else:
            players_id_list = range(n_agent_num[policy_i - 1], n_agent_num[policy_i])
        players_id.append(players_id_list)

        action_space_list = [g.get_single_action_space(player_id) for player_id in players_id_list]
        actions_space.append(action_space_list)

    return players_id, actions_space

def get_joint_action_eval(game, multi_part_agent_ids, policy_list, actions_spaces, all_observes, style_states):
    if len(policy_list) != len(game.agent_nums):
        error = "The number of models %d and the number of players %d have mismatched!" % (len(policy_list), len(game.agent_nums))
        raise Exception(error)

    joint_action = []
    for policy_i in range(len(policy_list)):

        if game.obs_type[policy_i] not in obs_type:
            raise Exception("Optional observation types: %s" % str(obs_type))

        agents_id_list = multi_part_agent_ids[policy_i]

        action_space_list = actions_spaces[policy_i]
        function_name = 'm%d' % policy_i

        style_state = style_states[policy_i]
        
        if policy_list[policy_i] in no_batch_agents:
            for i in range(len(agents_id_list)):
                agent_id = agents_id_list[i]
                a_obs = all_observes[agent_id]
                if 'lcmsp' in policy_list[policy_i] and 'mask' not in policy_list[policy_i]:
                    each = eval(function_name)(a_obs, action_space_list[i], game.is_act_continuous, style_state)
                else:
                    each = eval(function_name)(a_obs, action_space_list[i], game.is_act_continuous)
                # each = eval(function_name)(a_obs, action_space_list[i], game.is_act_continuous, style_state)
                joint_action.extend([each])
                # print(each)
        else:
            obs_batch = []
            for i in range(len(agents_id_list)):
                agent_id = agents_id_list[i]
                a_obs = all_observes[agent_id]
                obs_batch.append(a_obs)
            if len(obs_batch) > 0:
                if 'lcdsp' in policy_list[policy_i] and 'mask' not in policy_list[policy_i]:
                    each = eval(function_name)(obs_batch, action_space_list[i], game.is_act_continuous, style_state)
                else:
                    each = eval(function_name)(obs_batch, action_space_list[i], game.is_act_continuous)
                # each = eval(function_name)(obs_batch, action_space_list[i], game.is_act_continuous, style_state)
                joint_action.extend(each)
    return joint_action

def set_seed(g, env_name):
    if env_name.split("-")[0] in ['magent']:
        g.reset()
        seed = g.create_seed()
        g.set_seed(seed)

def natural_language_input_process(queue):
    root = tk.Tk()
    root.withdraw()  

    while True:
        user_input = simpledialog.askstring("Input box", "Please enter text (type 'exit' to finish):")
        
        if user_input == 'exit':
            queue.put(None)  
            break
        
        queue.put(user_input)

    root.quit()

def style_input_process(queue, data_dict):
    def submit():
        for key, entry in entries.items():
            value = entry.get()
            try:
                data_dict[key] = eval(value)
            except:
                data_dict[key] = value
        queue.put(data_dict)

    root = tk.Tk()
    root.title("Dictionary Editor")

    entries = {}
    for i, (key, value) in enumerate(data_dict.items()):
        label = tk.Label(root, text=key)
        label.grid(row=i, column=0)
        entry = tk.Entry(root)
        entry.insert(0, str(value))
        entry.grid(row=i, column=1)
        entries[key] = entry

    submit_button = tk.Button(root, text="Submit", command=submit)
    submit_button.grid(row=len(data_dict), column=0, columnspan=2)

    root.mainloop()

def run_game(g, env_name, multi_part_agent_ids, actions_spaces, policy_list, render_mode, style_states, agent_type, 
             queue, language_input=False, style_input=False, predictor=None):

    log_path = os.getcwd() + '/logs/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = get_logger(log_path, g.game_name, json_file=render_mode)
    set_seed(g, env_name)

    for i in range(len(policy_list)):
        if policy_list[i] not in get_valid_agents(agent_type):
            raise Exception("agent {} not valid!".format(policy_list[i]))

        file_path = os.path.dirname(os.path.abspath(__file__)) + "/{}/".format(agent_type) + policy_list[i] + "/submission.py"
        print(file_path)
        if not os.path.exists(file_path):
            raise Exception("file {} not exist!".format(file_path))

        import_path = '.'.join(file_path.split('/')[-3:])[:-3]
        function_name = 'm%d' % i
        if 'arg' in policy_list[i]:
            import_name = "my_controller_argmax"
        elif policy_list[i] not in no_batch_agents:
            import_name = "my_controller_batch"
        else:
            import_name = "my_controller"
        import_s = "from %s import %s as %s" % (import_path, import_name, function_name)
        print(import_s)
        exec(import_s, globals())

    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info = {"game_name": env_name,
                 "n_player": g.n_player,
                 "board_height": g.board_height if hasattr(g, "board_height") else None,
                 "board_width": g.board_width if hasattr(g, "board_width") else None,
                 "init_info": g.init_info,
                 "start_time": st,
                 "mode": "terminal",
                 "seed": g.seed if hasattr(g, "seed") else None,
                 "map_size": g.map_size if hasattr(g, "map_size") else None}

    steps = []
    all_observes = g.all_observes
     
    while not g.is_terminal():
        step = "step%d" % g.step_cnt

        if render_mode:
            if hasattr(g, "env_core"):
                if hasattr(g.env_core, "render"):
                    g.env_core.render()
                    
        if queue and not queue.empty():
            if language_input:
                user_input = queue.get()
                print(f"The user input text is: {user_input}")
                # instruction = natural_language_to_style(user_input)
                # factor = word_to_reward_factor(instruction)
                factor = predictor.predict(user_input)
                style_states[0].update(factor)
                print("The style parameters for the left-side team are now: ", style_states[0])
            if style_input:
                updated_dict = queue.get()
                print("Updated Dictionary:", updated_dict)
                style_states[0].update(updated_dict)
                print("The style parameters for the left-side team are now: ", style_states[0])
            
        info_dict = {"time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}
        style_states_list = [concate_style_state(style_states[0]), concate_style_state(style_states[1])]
        joint_act = get_joint_action_eval(
            g, multi_part_agent_ids, policy_list, actions_spaces, all_observes, style_states_list)
        all_observes, reward, done, info_before, info_after = g.step(joint_act)
        if render_mode:
            if hasattr(g, "render") and g.game_name == 'logistics_transportation':
                g.render()
        if env_name.split("-")[0] in ["magent"]:
            info_dict["joint_action"] = g.decode(joint_act)
        if info_before:
            info_dict["info_before"] = info_before
        info_dict["reward"] = reward
        if info_after:
            info_dict["info_after"] = info_after
        steps.append(info_dict)
        # print("step : {}, reward: {}".format(g.step_cnt, reward))
        # time.sleep(0.05)

    game_info["steps"] = steps
    game_info["winner"] = g.check_win()
    game_info["winner_information"] = g.won
    game_info["n_return"] = g.n_return
    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info["end_time"] = ed
    logs = json.dumps(game_info, ensure_ascii=False, cls=NpEncoder)
    logger.info(logs)

def get_valid_agents(agent_type):
    dir_path = os.path.join(os.path.dirname(__file__), '{}'.format(agent_type))
    return [f for f in os.listdir(dir_path) if f != "__pycache__"]

def get_style_state(factors):
    reward_factors = {}
    style_state = []
    for factor_name, factor_info in factors.items():
        lower_bound = factor_info['lower_bound']
        upper_bound = factor_info['upper_bound']
        fine_grit = factor_info['fine_grit']

        # Check if lower_bound and upper_bound are lists
        if factor_name == 'shot':
                reward_factors_list = [0.0] * len(lower_bound)
                random_index = 1
                
                reward_factors_list[random_index] = 1.0
                style_values_list[random_index] = 1.0
                    
                reward_factors[factor_name] = reward_factors_list
                style_state.extend(style_values_list)
        elif isinstance(lower_bound, list) and isinstance(upper_bound, list):
            # Handle the list case
            reward_factors_list = []
            style_values_list = []
            
            # NOTE: This method involves one-hot selecting an index within a list.
            reward_factors_list= [0.0] * len(lower_bound)
            random_index = random.randint(0, len(lower_bound) - 1)
            reward_factors_list[random_index] = 1.0
            style_values_list = reward_factors_list
            
            reward_factors[factor_name] = reward_factors_list
            style_state.extend(style_values_list)
        else:
            # Original handling for non-list bounds
            if fine_grit > 0:
                if lower_bound == upper_bound:
                    reward_value = lower_bound
                else:
                    grit_factor = np.linspace(lower_bound, upper_bound, int((upper_bound - lower_bound) / fine_grit) + 1)
                    reward_value = random.choice(grit_factor)
            else:
                reward_value = random.uniform(lower_bound, upper_bound)
            reward_factors[factor_name] = reward_value
            # Calculate style values for the non-list case
            style_value = (reward_value - lower_bound) / (upper_bound - lower_bound) if upper_bound != lower_bound else lower_bound / 2
            style_state.append(style_value)
        
    # NOTE: Fill the reserved extra style parameter slots with zeros.
    while len(style_state) < 40:
        style_state.append(0)
    
    return style_state, reward_factors

def concate_style_state(style_states):
    new_style_states = []
    for style, style_value in style_states.items():
        if isinstance(style_value, list):
            new_style_states.extend(style_value)
        else:
            new_style_states.append(style_value)
    while len(new_style_states) < 40:
        new_style_states.append(0)
    return new_style_states

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="lcdsp_5v5",
                        help="buildin_AI/lcdsp_5v5")
    parser.add_argument("--opponent", default="lcdsp_5v5",
                        help="buildin_AI/lcdsp_5v5")
    parser.add_argument("--env", default="football_5v5_malib",
                        help="football_5v5_malib")
    parser.add_argument("--agent_type", default="agents_5v5", help="agents_5v5")
    parser.add_argument("--language_input", default=False, help="True/False")
    parser.add_argument("--style_input", default=False, help="True/False")
    args = parser.parse_args()

    no_batch_agents = []
    game = make(args.env)
    
    # seed 
    th.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # whether to render
    render_mode = True
    policy_list = [args.my_ai, args.opponent]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, 'language_control'))

    if args.env == "football_5v5_malib":
        style_config = 'base_style_parameters\\style_5v5.json'
        sys.path.append(os.path.join(current_dir, 'language_control\\5v5'))
    with open(style_config) as f:
        style_factors = f.read()
    style_factors = json.loads(style_factors)
    style_state_1, style_state_2 = style_factors["style_state_1"], style_factors["style_state_2"]
    
    multi_part_agent_ids, actions_space = get_players_and_action_space_list(game)
    inital_obs_list = []
    ep_cnt = 0
    
    language_input = args.language_input
    style_input = args.style_input
    queue = None
    predictor = None
    if language_input:
        from natural_language_to_style import StyleInterpreterPredictor
        predictor = StyleInterpreterPredictor(model_path=os.path.join(current_dir, 'language_models', 'trained_model.pth'))  # 需要提供实际模型路径
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=natural_language_input_process, args=(queue,))
        p.start()
    if style_input:
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=style_input_process, args=(queue, style_state_1))
        p.start()
    try:
        while True:
            run_game(game, args.env, multi_part_agent_ids, actions_space,
                    policy_list, render_mode, [style_state_1, style_state_2], args.agent_type, 
                    queue=queue, language_input=language_input, style_input=style_input, predictor=predictor)
            game = make(args.env)
            ep_cnt += 1
    except KeyboardInterrupt:
        print("The main process received an interrupt signal and is shutting down...")
    finally:
        if language_input or style_input:
            p.terminate()
            p.join()
            print("The subprocess has terminated.")