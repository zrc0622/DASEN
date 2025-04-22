import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
from run import run

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def _get_config_name(params, arg_name):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        return config_name

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_agent_and_map_name(params, config_dict):
    # Parse the agent name
    agent_name = None
    for _i, _v in enumerate(params):
        if _v.startswith("--agent="):
            agent_name = _v.split("=")[1]
            del params[_i]  # remove the agent argument from params
            break
    
    if agent_name:
        config_dict['agent'] = agent_name  # Add agent name to config
    
    # Parse map_name and update ally_num and enemy_num
    for _i, _v in enumerate(params):
        if _v.startswith("--env-args.map_name="):
            map_name = _v.split("=")[1]
            del params[_i]  # remove the map_name argument from params
            break
    
    if 'map_name' in locals():
        if "m_vs_" in map_name:
            ally_num, enemy_num = map_name.split("m_vs_")
            config_dict['env_args']['ally_num'] = int(ally_num)
            config_dict['env_args']['enemy_num'] = int(enemy_num)
        elif "m" in map_name:
            num = int(map_name.split("m")[0])
            config_dict['env_args']['ally_num'] = num
            config_dict['env_args']['enemy_num'] = num


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    config_name = _get_config_name(params, "--total-config")
    with open(os.path.join(os.path.dirname(__file__), "config", f"{config_name}.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--alg-config", "algs")
    
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # Parse agent name and map name
    parse_agent_and_map_name(params, config_dict)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
