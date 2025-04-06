import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
import torch
import logging
from utils.utils import create_logger, copy_all_src
from utils.functions import seed_everything
# from PoppyCVRPTrainer import PoppyCVRPTrainer
from revised_poppy_cvrp_trainer import PoppyCVRPTrainer

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0  # $ CUDA_VISIBLE_DEVICES=0 nohup python -u train.py 2>&1 &

##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'norm': 'instance'
}

# optimizer_params = {
#     'optimizer': {
#         'lr': 1e-4,
#         'weight_decay': 1e-6,
#         'capturable': True
#     },
#     'scheduler': {
#         'milestones': [301, ],
#         'gamma': 0.1
#     }
# }

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6,
        'capturable': True
    },
    'scheduler': {
        'T_max': 500,
        'eta_min': 1e-6
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'seed': 3407,
    'phase': 2,  # 1 for pretraining, 2 for population training
    'method': 'poppy',
    'routing_model': False,
    'global_attack': True,
    'pretrain_epochs': 30500,
    'epochs': 500,
    # 'train_episodes': 10 * 1000,
    'train_episodes': 10 * 500,
    'num_expert': 3,  # Number of agents in population
    'train_batch_size': 64,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'general.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    # load checkpoint for phase 2
    'model_load': {
        'enable': False,  # enable loading population checkpoint
        'path': './result/saved_CVRP100_poppy',  # directory path of checkpoint
        'epoch': 100,  # epoch version to load
    },
    # load pretrain model for phase 1 or as base for phase 2
    'pretrain_load': {
        'enable': True,  # Enable loading pretrained model
        'path': '../../pretrained/POMO-CVRP100',  # Path to pretrained model
        'epoch': 30500,  # Epoch of pretrained model to load
    }
}

adv_params = {
    'eps_min': 1,
    'eps_max': 100,
    'num_steps': 1,
    'perturb_demand': False
}

logger_params = {
    'log_file': {
        'desc': 'train_poppy_cvrp',
        'filename': 'log.txt'
    }
}


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    seed_everything(trainer_params['seed'])

    print(f">> Starting Poppy Training - Phase {trainer_params['phase']}")
    print(f">> Number of experts: {trainer_params['num_expert']}")
    
    trainer = PoppyCVRPTrainer(
        env_params=env_params, 
        model_params=model_params, 
        optimizer_params=optimizer_params, 
        trainer_params=trainer_params, 
        adv_params=adv_params
    )

    copy_all_src(trainer.result_folder)
    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['pretrain_epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":
    main()