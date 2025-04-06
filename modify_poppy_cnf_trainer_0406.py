import os
import random
import pickle
import argparse
import torch
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
from CVRProblemDef import get_random_problems, generate_x_adv
from CVRP_baseline import solve_hgs_log, get_hgs_executable
from torch.optim import Adam as Optimizer
 # from torch.optim.lr_scheduler import MultiStepLR as Scheduler  # original scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR as Scheduler  # updated to cosine annealing

from generate_adv import generate_adv_dataset
from utils.utils import *
from utils.functions import *


class PoppyCVRPTrainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params, adv_params):
        """
        Initialize the Poppy trainer for CVRP problems.
        
        Args:
            env_params: Environment parameters
            model_params: Model architecture parameters
            optimizer_params: Optimizer settings
            trainer_params: Training control parameters
            adv_params: Adversarial example generation parameters
        """
        # Save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.adv_params = adv_params

        # Result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # CUDA setup
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device
        self.env_params['device'] = device
        self.model_params['device'] = device

        # Main Components
        self.num_expert = self.trainer_params['num_expert']
        self.env = Env(**self.env_params)
        
        # Setup for the current training phase
        self.phase = self.trainer_params.get('phase', 1)  # Default to phase 1
        
        if self.phase == 1:
            # Phase 1: Single model pre-training
            self.pre_model = Model(**self.model_params)
            self.pre_optimizer = Optimizer(self.pre_model.parameters(), **self.optimizer_params['optimizer'])
        else:
            # Phase 2: Population training with shared encoder
            # Initialize population from a pretrained model
            self._setup_population()
        
        # Restore checkpoint if needed
        self._restore_checkpoint()
        
        # Utility
        self.time_estimator = TimeEstimator()

    def _setup_population(self):
        """
        Setup the population of models with shared encoder for Phase 2.
        """
        # Load the pre-trained model
        pretrain_load = self.trainer_params['pretrain_load']
        if not pretrain_load['enable']:
            self.logger.error("Pretrained model must be provided for Phase 2")
            raise ValueError("Pretrained model must be provided for Phase 2")
            
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**pretrain_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        
        # Create population of models (all sharing the first model's encoder)
        self.models = []
        self.optimizers = []
        self.schedulers = []
        
        # First model is the complete model that contains both encoder and decoder
        self.models.append(Model(**self.model_params))
        self.models[0].load_state_dict(checkpoint['model_state_dict'])
        
        # For remaining models, create models and then share the encoder
        for i in range(1, self.num_expert):
            model = Model(**self.model_params)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Hook the model to share the encoder with the first model
            self._hook_encoder_sharing(model, self.models[0])
            
            self.models.append(model)
            
        # Create optimizers and schedulers for all models
        for model in self.models:
            optimizer = Optimizer(model.parameters(), **self.optimizer_params['optimizer'])
            scheduler = Scheduler(optimizer, **self.optimizer_params['scheduler'])
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
            
        self.logger.info(f"Set up population with {self.num_expert} models sharing encoder")
    
    def _hook_encoder_sharing(self, model, encoder_model):
        """
        Hook a model's pre_forward method to share its encoder with another model.
        
        Args:
            model: The model that will share its encoder
            encoder_model: The model containing the encoder to share
        """
        original_pre_forward = model.pre_forward
        
        def new_pre_forward(reset_state):
            # Use the encoder from the first model
            encoder_model.pre_forward(reset_state)
            model.encoded_nodes = encoder_model.encoded_nodes
            model.decoder.set_kv(model.encoded_nodes)
        
        model.pre_forward = new_pre_forward
    
    def _restore_checkpoint(self):
        """
        Restore from checkpoint based on the current phase.
        """
        self.start_epoch = 1
        
        if self.phase == 1:
            # For Phase 1, try to load a pretrained model
            pretrain_load = self.trainer_params['pretrain_load']
            if pretrain_load['enable']:
                checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**pretrain_load)
                checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
                self.pre_model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Pretrain model loaded successfully from {checkpoint_fullname}")
                
                # If there's result_log data, also load that
                if 'result_log' in checkpoint:
                    self.result_log.set_raw_data(checkpoint['result_log'])
        else:
            # For Phase 2, try to load a population checkpoint
            model_load = self.trainer_params['model_load']
            if model_load['enable']:
                checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
                checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
                
                model_state_dict = checkpoint['model_state_dict']
                optimizer_state_dict = checkpoint['optimizer_state_dict']
                
                for i in range(self.num_expert):
                    self.models[i].load_state_dict(model_state_dict[i])
                    self.optimizers[i].load_state_dict(optimizer_state_dict[i])
                    self.schedulers[i].last_epoch = model_load['epoch'] - 1
                
                self.start_epoch = 1 + model_load['epoch']
                self.result_log.set_raw_data(checkpoint['result_log'])
                self.logger.info(f"Population checkpoint loaded successfully from {checkpoint_fullname}")

    def run(self):
        """
        Main training loop - dispatches to the appropriate phase.
        """
        if self.phase == 1:
            self._run_phase_1()
        else:
            self._run_phase_2()
    
    def _run_phase_1(self):
        """
        Phase 1: Pre-training a single model on natural instances.
        """
        self.time_estimator.reset(self.start_epoch)
        
        for epoch in range(self.start_epoch, self.trainer_params['pretrain_epochs']+1):
            self.logger.info('=================================================================')
            
            # Train on natural instances and get the average loss
            avg_loss = self._train_one_epoch_phase_1(epoch)
            
            # Save the loss in result_log
            self.result_log.append('train_loss', epoch, avg_loss)
            
            # Log progress and save checkpoints
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                epoch, self.trainer_params['pretrain_epochs'])
            
            self.logger.info(f"Epoch {epoch}/{self.trainer_params['pretrain_epochs']}: "
                            f"Time Est.: Elapsed[{elapsed_time_str}], Remain[{remain_time_str}], "
                            f"Loss: {avg_loss:.6f}")
            
            # Save visualization for loss
            if epoch > 1:
                self.logger.info("Saving log_image for loss")
                image_prefix = f'{self.result_folder}/latest'
                util_save_log_image_with_label(
                    image_prefix, 
                    self.trainer_params['logging']['log_image_params_2'], 
                    self.result_log, 
                    labels=['train_loss']
                )
            
            # Save checkpoint periodically
            if epoch % self.trainer_params['logging']['model_save_interval'] == 0 or \
               epoch == self.trainer_params['pretrain_epochs']:
                self.logger.info("Saving pretrained model")
                
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.pre_model.state_dict(),
                    'optimizer_state_dict': self.pre_optimizer.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                
                torch.save(checkpoint_dict, f'{self.result_folder}/checkpoint-{epoch}.pt')
        
        self.logger.info("*** Phase 1 Pre-training Completed ***")
    
    # def _run_phase_2(self):
    #     """
    #     Phase 2: Training a population of models with the Winner-Takes-All approach.
    #     """
    #     self.time_estimator.reset(self.start_epoch)
        
    #     for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
    #         self.logger.info('=================================================================')
            
    #         # Train the population and get the average loss
    #         avg_loss = self._train_one_epoch_phase_2(epoch)
            
    #         # Save the loss in result_log
    #         self.result_log.append('train_loss', epoch, avg_loss)
            
    #         # Apply scheduler step to all models
    #         for scheduler in self.schedulers:
    #             scheduler.step()
            
    #         # [Old] Validation on natural and adversarial datasets
    #         # [Old] dir = "../../data/CVRP"
    #         # [Old] paths = ["cvrp50_uniform.pkl", "adv_cvrp50_uniform.pkl"]
    #         # [Old] val_episodes, score_list, gap_list = 1000, [], []
    #         # [Old] 
    #         # [Old] for path in paths:
    #         # [Old]     score, gap = self._val_and_stat(dir, path, batch_size=500, val_episodes=val_episodes)
    #         # [Old]     score_list.append(score)
    #         # [Old]     gap_list.append(gap)
    #         # [Old] 
    #         # [Old] # Fixed: Log each score and gap separately with a unique key
    #         # [Old] self.result_log.append('val_score_natural', epoch, score_list[0])
    #         # [Old] self.result_log.append('val_score_adversarial', epoch, score_list[1])
    #         # [Old] self.result_log.append('val_gap_natural', epoch, gap_list[0])
    #         # [Old] self.result_log.append('val_gap_adversarial', epoch, gap_list[1])
            
    #         # [New] Additional validation on uniform, fixed adv, and dynamic adv datasets
    #         dir = "../../data/CVRP"
            
    #         # 1. Test on uniform (clean) dataset
    #         uniform_path = "cvrp100_uniform.pkl"
    #         start_time = time.time()
    #         score_uniform, gap_uniform = self._val_and_stat(dir, uniform_path, batch_size=500, val_episodes=1000)
    #         inference_time = time.time() - start_time
    #         self.logger.info(f"Inference Time on {uniform_path}: {inference_time:.2f}s")
    #         self.result_log.append(f'inference_time_{uniform_path.split(".")[0]}', epoch, inference_time)
    #         self.logger.info(f">> Val Result on {uniform_path}: SCORE={score_uniform:.4f}, GAP={gap_uniform:.4f}%")
            
    #         # 2. Test on fixed adversarial dataset
    #         fixed_adv_path = "adv_cvrp100_uniform.pkl"
    #         start_time = time.time()
    #         score_fixed_adv, gap_fixed_adv = self._val_and_stat(dir, fixed_adv_path, batch_size=500, val_episodes=1000)
    #         inference_time = time.time() - start_time
    #         self.logger.info(f"Inference Time on {fixed_adv_path}: {inference_time:.2f}s")
    #         self.result_log.append(f'inference_time_{fixed_adv_path.split(".")[0]}', epoch, inference_time)
    #         self.logger.info(f">> Val Result on {fixed_adv_path}: SCORE={score_fixed_adv:.4f}, GAP={gap_fixed_adv:.4f}%")
            
    #         # 3. Dynamically generate adversarial dataset based on current model
    #         # [Fix] Correct dynamic adversarial dataset generation: all experts attack + random sampling
    #         data = load_dataset(os.path.join(dir, uniform_path))[:1000]
    #         depot_xy = torch.Tensor([i[0] for i in data])
    #         node_xy = torch.Tensor([i[1] for i in data])
    #         node_demand = torch.Tensor([i[2] for i in data])
    #         capacity = torch.Tensor([i[3] for i in data])
    #         node_demand = node_demand / capacity.view(-1, 1)
    #         test_data = (depot_xy, node_xy, node_demand)
            
    #         # Let each model generate adversarial examples
    #         all_adv_node_xy = []
    #         all_adv_node_demand = []
            
    #         for model in self.models:
    #             adv_data = generate_adv_dataset(
    #                 model,
    #                 test_data,
    #                 eps_min=self.adv_params['eps_min'],
    #                 eps_max=self.adv_params['eps_max'],
    #                 num_steps=self.adv_params['num_steps'],
    #                 perturb_demand=self.adv_params['perturb_demand']
    #             )
    #             all_adv_node_xy.append(adv_data[1])
    #             all_adv_node_demand.append(adv_data[2])
            
    #         # Concatenate adversarial samples from all experts
    #         all_adv_node_xy = torch.cat(all_adv_node_xy, dim=0)
    #         all_adv_node_demand = torch.cat(all_adv_node_demand, dim=0)
    #         repeated_depot_xy = depot_xy.repeat(self.num_expert, 1, 1)
            
    #         # Randomly sample 1000 adversarial instances
    #         perm = torch.randperm(all_adv_node_xy.size(0))[:1000]
    #         sampled_depot_xy = repeated_depot_xy[perm]
    #         sampled_node_xy = all_adv_node_xy[perm]
    #         sampled_node_demand = all_adv_node_demand[perm]
            
    #         # === Dynamic adversarial dataset generation is skipped; using pre-generated datasets ===
    #         # The following dynamic generation code is commented out:
    #         # save_path = os.path.join(dir, "current_adv_cvrp100_uniform.pkl")
    #         # sampled_capacity = torch.full((sampled_depot_xy.size(0),), 50)  # CVRP100的标准容量50
    #         # adv_data = (sampled_depot_xy, sampled_node_xy, sampled_node_demand, sampled_capacity)
    #         # save_dataset(list(zip(adv_data[0].tolist(), adv_data[1].tolist(), adv_data[2].tolist(), adv_data[3].tolist())), save_path)
    #         # from utils.utils import save_dataset_as_vrplib  # 确保utils里面有save vrp的方法
    #         # save_dataset_as_vrplib(list(zip(adv_data[0].tolist(), adv_data[1].tolist(), adv_data[2].tolist(), adv_data[3].tolist())),
    #         #                        save_path.replace(".pkl", ""))
    #         # params = argparse.ArgumentParser()
    #         # params.cpus, params.n, params.progress_bar_mininterval = None, None, 0.1
    #         # executable = get_hgs_executable()
    #         # vrp_dir = save_path.replace(".pkl", "")
    #         # def run_func(filename):
    #         #     return solve_hgs_log(executable, filename, runs=1, disable_cache=True)
    #         # filenames = [os.path.join(vrp_dir, f) for f in sorted(os.listdir(vrp_dir)) if f.endswith('.vrp')]
    #         # results, _ = run_all_in_pool(run_func, "./HGS_result", filenames, params, use_multiprocessing=False)
    #         # os.system("rm -rf ./HGS_result")
    #         # results = [(i[0], i[1]) for i in results]
    #         # save_dataset(results, os.path.join(dir, "hgs_current_adv_cvrp100_uniform.pkl"))
            
    #         # 4. Test on pre-generated adversarial dataset
    #         dynamic_adv_path = "adv_cvrp100_uniform.pkl"
    #         start_time = time.time()
    #         score_dynamic_adv, gap_dynamic_adv = self._val_and_stat(dir, dynamic_adv_path, batch_size=500, val_episodes=1000)
    #         inference_time = time.time() - start_time
    #         self.logger.info(f"Inference Time on {dynamic_adv_path}: {inference_time:.2f}s")
    #         self.result_log.append(f'inference_time_{dynamic_adv_path.split(".")[0]}', epoch, inference_time)
    #         self.logger.info(f">> Val Result on {dynamic_adv_path}: SCORE={score_dynamic_adv:.4f}, GAP={gap_dynamic_adv:.4f}%")
            
    #         # 5. Log new validation results
    #         self.result_log.append('val_score_uniform', epoch, score_uniform)
    #         self.result_log.append('val_gap_uniform', epoch, gap_uniform)
    #         self.result_log.append('val_score_fixed_adv', epoch, score_fixed_adv)
    #         self.result_log.append('val_gap_fixed_adv', epoch, gap_fixed_adv)
    #         self.result_log.append('val_score_dynamic_adv', epoch, score_dynamic_adv)
    #         self.result_log.append('val_gap_dynamic_adv', epoch, gap_dynamic_adv)
            
    #         # Log progress and save checkpoints
    #         elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
    #             epoch, self.trainer_params['epochs'])
            
    #         self.logger.info(f"Epoch {epoch}/{self.trainer_params['epochs']}: "
    #                         f"Time Est.: Elapsed[{elapsed_time_str}], Remain[{remain_time_str}], "
    #                         f"Loss: {avg_loss:.6f}")
            
    #         # Save visualization
    #         if epoch > 1:
    #             self.logger.info("Saving log_images")
    #             image_prefix = f'{self.result_folder}/latest'
                
    #             # Fixed: Plot each metric separately
    #             # [Old] util_save_log_image_with_label(
    #             # [Old]     image_prefix + "_natural_score", 
    #             # [Old]     self.trainer_params['logging']['log_image_params_1'], 
    #             # [Old]     self.result_log, 
    #             # [Old]     labels=['val_score_natural']
    #             # [Old] )
    #             # [Old] 
    #             # [Old] util_save_log_image_with_label(
    #             # [Old]     image_prefix + "_adversarial_score", 
    #             # [Old]     self.trainer_params['logging']['log_image_params_1'], 
    #             # [Old]     self.result_log, 
    #             # [Old]     labels=['val_score_adversarial']
    #             # [Old] )
    #             # [Old] 
    #             # [Old] util_save_log_image_with_label(
    #             # [Old]     image_prefix + "_natural_gap", 
    #             # [Old]     self.trainer_params['logging']['log_image_params_1'], 
    #             # [Old]     self.result_log, 
    #             # [Old]     labels=['val_gap_natural']
    #             # [Old] )
    #             # [Old] 
    #             # [Old] util_save_log_image_with_label(
    #             # [Old]     image_prefix + "_adversarial_gap", 
    #             # [Old]     self.trainer_params['logging']['log_image_params_1'], 
    #             # [Old]     self.result_log, 
    #             # [Old]     labels=['val_gap_adversarial']
    #             # [Old] )
                
    #             # [Old] Save train_loss visualization
    #             # [Old] util_save_log_image_with_label(
    #             # [Old]     image_prefix + "_loss", 
    #             # [Old]     self.trainer_params['logging']['log_image_params_2'], 
    #             # [Old]     self.result_log, 
    #             # [Old]     labels=['train_loss']
    #             # [Old] )
                
    #             # [New] Save uniform score visualization
    #             util_save_log_image_with_label(
    #                 image_prefix + "_uniform_score",
    #                 self.trainer_params['logging']['log_image_params_1'],
    #                 self.result_log,
    #                 labels=['val_score_uniform']
    #             )
                
    #             # [New] Save fixed adv score visualization
    #             util_save_log_image_with_label(
    #                 image_prefix + "_fixed_adv_score",
    #                 self.trainer_params['logging']['log_image_params_1'],
    #                 self.result_log,
    #                 labels=['val_score_fixed_adv']
    #             )
                
    #             # [New] Save dynamic adv score visualization
    #             util_save_log_image_with_label(
    #                 image_prefix + "_dynamic_adv_score",
    #                 self.trainer_params['logging']['log_image_params_1'],
    #                 self.result_log,
    #                 labels=['val_score_dynamic_adv']
    #             )
                
    #             # [New] Save uniform gap visualization
    #             util_save_log_image_with_label(
    #                 image_prefix + "_uniform_gap",
    #                 self.trainer_params['logging']['log_image_params_1'],
    #                 self.result_log,
    #                 labels=['val_gap_uniform']
    #             )
                
    #             # [New] Save fixed adv gap visualization
    #             util_save_log_image_with_label(
    #                 image_prefix + "_fixed_adv_gap",
    #                 self.trainer_params['logging']['log_image_params_1'],
    #                 self.result_log,
    #                 labels=['val_gap_fixed_adv']
    #             )
                
    #             # [New] Save dynamic adv gap visualization
    #             util_save_log_image_with_label(
    #                 image_prefix + "_dynamic_adv_gap",
    #                 self.trainer_params['logging']['log_image_params_1'],
    #                 self.result_log,
    #                 labels=['val_gap_dynamic_adv']
    #             )
                
    #             # [New] Save train loss visualization
    #             util_save_log_image_with_label(
    #                 image_prefix + "_loss",
    #                 self.trainer_params['logging']['log_image_params_2'],
    #                 self.result_log,
    #                 labels=['train_loss']
    #             )
            
    #         # Save checkpoint
    #         all_done = (epoch == self.trainer_params['epochs'])
    #         model_save_interval = self.trainer_params['logging']['model_save_interval']
            
    #         if all_done or (epoch % model_save_interval) == 0:
    #             self.logger.info("Saving trained models")
                
    #             checkpoint_dict = {
    #                 'epoch': epoch,
    #                 'model_state_dict': [model.state_dict() for model in self.models],
    #                 'optimizer_state_dict': [optimizer.state_dict() for optimizer in self.optimizers],
    #                 'scheduler_state_dict': [scheduler.state_dict() for scheduler in self.schedulers],
    #                 'result_log': self.result_log.get_raw_data()
    #             }
                
    #             torch.save(checkpoint_dict, f'{self.result_folder}/checkpoint-{epoch}.pt')
            
    #         if all_done:
    #             self.logger.info("*** Phase 2 Training Completed ***")
    #             # [New] Collect all validation logs into lists
    #             val_score_uniform = self.result_log.data['val_score_uniform']
    #             val_gap_uniform = self.result_log.data['val_gap_uniform']
                
    #             val_score_fixed_adv = self.result_log.data['val_score_fixed_adv']
    #             val_gap_fixed_adv = self.result_log.data['val_gap_fixed_adv']
                
    #             val_score_dynamic_adv = self.result_log.data['val_score_dynamic_adv']
    #             val_gap_dynamic_adv = self.result_log.data['val_gap_dynamic_adv']
                
    #             self.logger.info(f"Collected Validation Results:")
    #             self.logger.info(f"val_score_uniform: {val_score_uniform}")
    #             self.logger.info(f"val_gap_uniform: {val_gap_uniform}")
    #             self.logger.info(f"val_score_fixed_adv: {val_score_fixed_adv}")
    #             self.logger.info(f"val_gap_fixed_adv: {val_gap_fixed_adv}")
    #             self.logger.info(f"val_score_dynamic_adv: {val_score_dynamic_adv}")
    #             self.logger.info(f"val_gap_dynamic_adv: {val_gap_dynamic_adv}")
    
    def _generate_cur_adv(self, nat_data):
        """
            Note: nat_data should include depot_xy, node_xy, unnormalized node_demand and capacity,
            since we need to save data to the file system.
        """
        # generate adv examples based on current models
        depot_xy, node_xy, ori_node_demand, capacity = nat_data
        node_demand = ori_node_demand / capacity.view(-1, 1)
        data = (depot_xy, node_xy, node_demand)
        adv_node_xy = torch.zeros(0, data[1].size(1), 2)
        
        for i in range(self.num_expert):
            _, node, _ = generate_adv_dataset(
                self.models[i], 
                data, 
                eps_min=self.adv_params['eps_min'], 
                eps_max=self.adv_params['eps_max'], 
                num_steps=self.adv_params['num_steps'], 
                perturb_demand=self.adv_params['perturb_demand']
            )
            adv_node_xy = torch.cat((adv_node_xy, node), dim=0)
            
        adv_data = (
            torch.cat([depot_xy] * self.num_expert, dim=0), 
            adv_node_xy, 
            torch.cat([ori_node_demand] * self.num_expert, dim=0), 
            torch.cat([capacity] * self.num_expert, dim=0)
        )
        
        with open("./adv_tmp.pkl", "wb") as f:
            pickle.dump(
                list(zip(
                    adv_data[0].tolist(), 
                    adv_data[1].tolist(), 
                    adv_data[2].tolist(), 
                    adv_data[3].tolist()
                )), 
                f, 
                pickle.HIGHEST_PROTOCOL
            )  # [(depot_xy, node_xy, node_demand, capacity), ...]

        # obtain (sub-)opt solution using HGS
        params = argparse.ArgumentParser()
        params.cpus, params.n, params.progress_bar_mininterval = None, None, 0.1
        dataset = [attr.cpu().tolist() for attr in adv_data]
        dataset = [
            (dataset[0][i][0], dataset[1][i], [int(d) for d in dataset[2][i]], int(dataset[3][i])) 
            for i in range(adv_data[0].size(0))
        ]
        executable = get_hgs_executable()
        
        def run_func(args):
            return solve_hgs_log(executable, *args, runs=1, disable_cache=True)  # otherwise it directly loads data from dir
            
        results, _ = run_all_in_pool(run_func, "./HGS_result", dataset, params, use_multiprocessing=False)
        os.system("rm -rf ./HGS_result")
        results = [(i[0], i[1]) for i in results]
        save_dataset(results, "./hgs_adv_tmp.pkl")

    def _run_phase_2_modified(self):
        """
        Phase 2: Training a population of models with the Winner-Takes-All approach.
        Modified to process Fixed Adv and Adv like in CVRPTrainer.py
        """
        self.time_estimator.reset(self.start_epoch)
        
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            
            # Train the population and get the average loss
            avg_loss = self._train_one_epoch_phase_2(epoch)
            
            # Save the loss in result_log
            self.result_log.append('train_loss', epoch, avg_loss)
            
            # Apply scheduler step to all models
            for scheduler in self.schedulers:
                scheduler.step()
            
            # Validation on natural and adversarial datasets
            dir = "../../data/CVRP"
            paths = ["cvrp100_uniform.pkl", "adv_cvrp100_uniform.pkl"]
            val_episodes, score_list, gap_list = 1000, [], []
            
            # Generate dynamic adversarial dataset based on the status of current model
            data = load_dataset(os.path.join(dir, paths[0]), disable_print=True)[: val_episodes]
            depot_xy = torch.Tensor([i[0] for i in data])
            node_xy = torch.Tensor([i[1] for i in data])
            ori_node_demand = torch.Tensor([i[2] for i in data])
            capacity = torch.Tensor([i[3] for i in data])
            self._generate_cur_adv((depot_xy, node_xy, ori_node_demand, capacity))

            # Evaluate on uniform and fixed adversarial datasets
            for path in paths:
                score, gap = self._val_and_stat(dir, path, batch_size=500, val_episodes=val_episodes)
                score_list.append(score)
                gap_list.append(gap)
                
            # Evaluate on dynamically generated adversarial dataset
            score, gap = self._val_and_stat("./", "adv_tmp.pkl", batch_size=500, val_episodes=val_episodes * self.num_expert)
            score_list.append(score)
            gap_list.append(gap)
            
            # Save performance metrics to result_log
            self.result_log.append('val_score_uniform', epoch, score_list[0])
            self.result_log.append('val_score_fixed_adv', epoch, score_list[1])
            self.result_log.append('val_score_dynamic_adv', epoch, score_list[2])
            self.result_log.append('val_gap_uniform', epoch, gap_list[0])
            self.result_log.append('val_gap_fixed_adv', epoch, gap_list[1])
            self.result_log.append('val_gap_dynamic_adv', epoch, gap_list[2])
            
            # Log progress and save checkpoints
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                epoch, self.trainer_params['epochs'])
            
            self.logger.info(f"Epoch {epoch}/{self.trainer_params['epochs']}: "
                            f"Time Est.: Elapsed[{elapsed_time_str}], Remain[{remain_time_str}], "
                            f"Loss: {avg_loss:.6f}")
            
            # Save visualizations
            if epoch > 1:
                self.logger.info("Saving log_images")
                image_prefix = f'{self.result_folder}/latest'
                
                # Save visualizations for different metrics
                util_save_log_image_with_label(
                    image_prefix + "_uniform_score",
                    self.trainer_params['logging']['log_image_params_1'],
                    self.result_log,
                    labels=['val_score_uniform']
                )
                
                util_save_log_image_with_label(
                    image_prefix + "_fixed_adv_score",
                    self.trainer_params['logging']['log_image_params_1'],
                    self.result_log,
                    labels=['val_score_fixed_adv']
                )
                
                util_save_log_image_with_label(
                    image_prefix + "_dynamic_adv_score",
                    self.trainer_params['logging']['log_image_params_1'],
                    self.result_log,
                    labels=['val_score_dynamic_adv']
                )
                
                # Save loss visualization
                util_save_log_image_with_label(
                    image_prefix + "_loss",
                    self.trainer_params['logging']['log_image_params_2'],
                    self.result_log,
                    labels=['train_loss']
                )
            
            # Save checkpoint
            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained models")
                
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': [model.state_dict() for model in self.models],
                    'optimizer_state_dict': [optimizer.state_dict() for optimizer in self.optimizers],
                    'scheduler_state_dict': [scheduler.state_dict() for scheduler in self.schedulers],
                    'result_log': self.result_log.get_raw_data()
                }
                
                torch.save(checkpoint_dict, f'{self.result_folder}/checkpoint-{epoch}.pt')
            
            if all_done:
                self.logger.info("*** Phase 2 Training Completed ***")

    def _train_one_epoch_phase_1(self, epoch):
        """
        Train the single model for one epoch in Phase 1.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            float: Average loss for this epoch
        """
        episode = 0
        train_num_episode = self.trainer_params['train_episodes']
        total_loss = 0
        num_batches = 0
        
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            
            # Generate natural instances
            nat_data = get_random_problems(batch_size, self.env_params['problem_size'])
            depot_xy, node_xy, node_demand, capacity = nat_data
            node_demand = node_demand / capacity.view(-1, 1)
            nat_data = (depot_xy, node_xy, node_demand)
            
            # Forward pass
            score, loss = self._train_one_batch(self.pre_model, nat_data)
            avg_score, avg_loss = score.mean().item(), loss.mean()
            
            # Track loss
            total_loss += avg_loss.item()
            num_batches += 1
            
            # Backward pass
            self.pre_optimizer.zero_grad()
            avg_loss.backward()
            self.pre_optimizer.step()
            
            episode += batch_size
        
        # Calculate average loss for the epoch
        epoch_avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Log training progress
        self.logger.info(f'Epoch {epoch}: Train ({100.0 * episode / train_num_episode:.1f}%), Loss: {epoch_avg_loss:.6f}')
        
        return epoch_avg_loss
    
    def _train_one_epoch_phase_2(self, epoch):
        """
        Train the population for one epoch in Phase 2 using the Poppy approach.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            float: Average loss for this epoch
        """
        episode = 0
        train_num_episode = self.trainer_params['train_episodes']
        total_loss = 0
        num_batches = 0
        
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            
            # Generate natural instances
            nat_data = get_random_problems(batch_size, self.env_params['problem_size'])
            depot_xy, node_xy, node_demand, capacity = nat_data
            node_demand = node_demand / capacity.view(-1, 1)
            nat_data = (depot_xy, node_xy, node_demand)
            
            # Generate adversarial examples (inner maximization)
            eps = random.sample(range(self.adv_params['eps_min'], self.adv_params['eps_max'] + 1), 1)[0]
            all_data = nat_data
            
            # 1. Generate adversarial examples by each expert (local)
            for i in range(self.num_expert):
                depot, node, demand = generate_x_adv(
                    self.models[i], 
                    nat_data, 
                    eps=eps, 
                    num_steps=self.adv_params['num_steps'], 
                    perturb_demand=self.adv_params['perturb_demand']
                )
                all_data = (
                    torch.cat((all_data[0], depot), dim=0),
                    torch.cat((all_data[1], node), dim=0),
                    torch.cat((all_data[2], demand), dim=0)
                )
            
            # 2. Collaborate to generate adversarial examples (global)
            if self.trainer_params['global_attack']:
                data = nat_data
                for _ in range(self.adv_params['num_steps']):
                    scores = torch.zeros(batch_size, 0)
                    adv_depot = torch.zeros(0, 1, 2)
                    adv_node = torch.zeros(0, data[1].size(1), 2)
                    adv_demand = torch.zeros(0, data[2].size(1))
                    
                    for k in range(self.num_expert):
                        _, score = self._fast_val(self.models[k], data=data, aug_factor=1, eval_type="softmax")
                        scores = torch.cat((scores, score.unsqueeze(1)), dim=1)
                    
                    _, id = scores.min(1)
                    for k in range(self.num_expert):
                        mask = (id == k)
                        if mask.sum() < 1:
                            continue
                        
                        depot, node, demand = generate_x_adv(
                            self.models[k],
                            (data[0][mask], data[1][mask], data[2][mask]),
                            eps=eps,
                            num_steps=1,
                            perturb_demand=self.adv_params['perturb_demand']
                        )
                        adv_depot = torch.cat((adv_depot, depot), dim=0)
                        adv_node = torch.cat((adv_node, node), dim=0)
                        adv_demand = torch.cat((adv_demand, demand), dim=0)
                    
                    data = (adv_depot, adv_node, adv_demand)
                
                all_data = (
                    torch.cat((all_data[0], data[0]), dim=0),
                    torch.cat((all_data[1], data[1]), dim=0),
                    torch.cat((all_data[2], data[2]), dim=0)
                )
            
            # [New] Shuffle all_data before training
            perm = torch.randperm(all_data[0].size(0))
            all_data = (
                all_data[0][perm],
                all_data[1][perm],
                all_data[2][perm]
            )
            # Winner-Takes-All training (Poppy approach)
            batch_loss = self._poppy_update(all_data)
            
            # Track loss
            total_loss += batch_loss
            num_batches += 1
            
            episode += batch_size
        
        # Calculate average loss for the epoch
        epoch_avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Log training progress
        self.logger.info(f'Epoch {epoch}: Train ({100.0 * episode / train_num_episode:.1f}%), Loss: {epoch_avg_loss:.6f}')
        
        return epoch_avg_loss
    
    def _poppy_update(self, data):
        """
        Update the population using the Poppy Winner-Takes-All approach.
        Only train the best-performing agent on each instance.
        
        Args:
            data: Tuple of (depot_xy, node_xy, node_demand) containing problem instances
            
        Returns:
            float: Average loss across all trained agents
        """
        # Get performance of each model on the provided instances
        scores = torch.zeros(data[0].size(0), 0)
        
        for k in range(self.num_expert):
            _, score = self._fast_val(self.models[k], data=data, aug_factor=1, eval_type="softmax")
            scores = torch.cat((scores, score.unsqueeze(1)), dim=1)
        
        # For each instance, find the best and second-best agent
        best_scores, best_indices = scores.min(dim=1)
        
        # Get second-best scores by masking out the best score for each instance
        masked_scores = scores.clone()
        batch_indices = torch.arange(scores.shape[0], device=scores.device)
        masked_scores[batch_indices, best_indices] = float('inf')
        second_best_scores = masked_scores.min(dim=1)[0]
        
        # Calculate advantages: best_score - second_best_score
        advantages = second_best_scores - best_scores  # Larger is better for advantage
        
        # Track total loss for all agents
        total_agent_loss = 0
        num_trained_agents = 0
        
        # Group instances by best agent
        for agent_idx in range(self.num_expert):
            # Get instances where this agent was the best
            mask = (best_indices == agent_idx)
            
            if mask.sum() == 0:
                continue  # Skip if no instances for this agent
            
            # Get the data for this agent
            agent_data = (data[0][mask], data[1][mask], data[2][mask])
            agent_advantages = advantages[mask]
            
            # Train this agent only on its best instances
            agent_loss = self._train_agent_with_advantage(
                agent_idx, 
                agent_data,
                agent_advantages
            )
            
            total_agent_loss += agent_loss
            num_trained_agents += 1
        
        # Return average loss across all trained agents
        avg_loss = total_agent_loss / num_trained_agents if num_trained_agents > 0 else 0
        return avg_loss
    
    def _train_agent_with_advantage(self, agent_idx, data, advantages):
        """
        Train a single agent using the Winner-Takes-All advantage.
        
        Args:
            agent_idx: Index of the agent to train
            data: Problem instances for this agent
            advantages: The advantage values for each instance
            
        Returns:
            float: Loss value for this agent
        """
        model = self.models[agent_idx]
        model.train()
        
        batch_size = data[0].size(0)
        self.env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = self.env.reset()
        model.pre_forward(reset_state)
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        
        # POMO Rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        
        # Get log probabilities for each trajectory
        log_prob = prob_list.log().sum(dim=2)  # (batch, pomo)
        
        # Select the best trajectory for each instance
        max_reward_values, max_reward_indices = reward.max(dim=1)
        
        # Extract log probabilities of the best trajectories
        batch_indices = torch.arange(batch_size, device=reward.device)
        best_log_probs = log_prob[batch_indices, max_reward_indices]
        
        # Compute loss using the advantages (Winner-Takes-All)
        loss = -(advantages * best_log_probs).mean()
        loss_value = loss.item()
        
        # Backward pass
        self.optimizers[agent_idx].zero_grad()
        loss.backward()
        self.optimizers[agent_idx].step()
        
        return loss_value
    
    def _train_one_batch(self, model, data):
        """
        Train a single model on one batch of data using REINFORCE.
        
        Args:
            model: The model to train
            data: Problem instances
            
        Returns:
            score: Batch scores
            loss: Batch losses per instance
        """
        model.train()
        batch_size = data[0].size(0)
        self.env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = self.env.reset()
        model.pre_forward(reset_state)
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        
        # POMO Rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        
        # Compute advantage
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        
        # Compute log probabilities
        log_prob = prob_list.log().sum(dim=2)
        
        # Compute loss
        loss = -advantage * log_prob
        
        # Compute score (negative tour length)
        max_pomo_reward, _ = reward.max(dim=1)
        
        return -max_pomo_reward.float().detach(), loss.mean(1)
    
    def _fast_val(self, model, data=None, path=None, offset=0, val_episodes=1000, aug_factor=1, eval_type="argmax"):
        """
        Perform a fast evaluation of a model on a set of problem instances.
        
        Args:
            model: The model to evaluate
            data: Problem instances (if None, instances are loaded from path)
            path: Path to load instances from (if data is None)
            offset: Offset for loading instances from path
            val_episodes: Number of validation episodes
            aug_factor: Augmentation factor
            eval_type: Evaluation type (argmax or softmax)
            
        Returns:
            no_aug_score: Scores without augmentation
            aug_score: Scores with augmentation
        """
        if data is None:
            data = load_dataset(path, disable_print=True)[offset: offset + val_episodes]
            depot_xy = torch.Tensor([i[0] for i in data])
            node_xy = torch.Tensor([i[1] for i in data])
            node_demand = torch.Tensor([i[2] for i in data])
            capacity = torch.Tensor([i[3] for i in data])
            node_demand = node_demand / capacity.view(-1, 1)
            data = (depot_xy, node_xy, node_demand)
        
        env = Env(**{
            'problem_size': data[1].size(1), 
            'pomo_size': data[1].size(1), 
            'device': self.device
        })
        batch_size = data[0].size(0)
        
        model.eval()
        model.set_eval_type(eval_type)
        
        with torch.no_grad():
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)
        
        # Process results
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        max_pomo_reward, _ = aug_reward.max(dim=2)
        no_aug_score = -max_pomo_reward[0, :].float()
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)
        aug_score = -max_aug_pomo_reward.float()
        
        return no_aug_score, aug_score
    
    def _val_and_stat(self, dir, val_path, batch_size=500, val_episodes=1000):
        """
        Evaluate all models in the population and calculate statistics.
        
        Args:
            dir: Directory containing the datasets
            val_path: Path to the validation dataset
            batch_size: Batch size for evaluation
            val_episodes: Number of validation episodes
            
        Returns:
            tuple: (moe_aug_score_avg, moe_aug_gap_avg) - Mean scores and gaps for the population
        """
        import time
        start_time = time.time()
        # Use the problem size from env_params
        problem_size = self.env_params['problem_size']
    
        no_aug_score_list, aug_score_list = [], []
        no_aug_gap_list, aug_gap_list = [], []
        
        # Ensure no_aug_scores and aug_scores are created on the same device as the model
        no_aug_scores = torch.zeros(val_episodes, 0, device=self.device)
        aug_scores = torch.zeros(val_episodes, 0, device=self.device)
        
        # Load optimal solutions with dynamic adversarial test fix
        opt_filename = f"hgs_{val_path}"
        if val_path.startswith("current_adv"):
            opt_filename = "hgs_cvrp100_uniform.pkl"  # Dynamic Adv uses clean uniform's HGS optimal

        # opt_sol = load_dataset(os.path.join(dir, opt_filename), disable_print=True)[:val_episodes]
        # opt_sol = [i[0] for i in opt_sol]
        opt_filename = f"hgs_{val_path}"
        if val_path.startswith("current_adv"):
            opt_filename = "hgs_adv_cvrp100_uniform.pkl"  # 动态adv，直接用固定adv的最优解
        opt_sol = load_dataset(os.path.join(dir, opt_filename), disable_print=True)[:val_episodes]
        opt_sol = [i[0] for i in opt_sol]
        # Evaluate each model
        for i in range(self.num_expert):
            episode = 0
            no_aug_score = torch.zeros(0, device=self.device)
            aug_score = torch.zeros(0, device=self.device)
            
            while episode < val_episodes:
                remaining = val_episodes - episode
                bs = min(batch_size, remaining)
                
                no_aug, aug = self._fast_val(
                    self.models[i],
                    path=os.path.join(dir, val_path),
                    offset=episode,
                    val_episodes=bs,
                    aug_factor=8,
                    eval_type="argmax"
                )
                
                no_aug_score = torch.cat((no_aug_score, no_aug), dim=0)
                aug_score = torch.cat((aug_score, aug), dim=0)
                episode += bs
            
            # Store results for this model
            no_aug_score_list.append(round(no_aug_score.mean().item(), 4))
            aug_score_list.append(round(aug_score.mean().item(), 4))
            
            # Concatenate on the same device
            no_aug_scores = torch.cat((no_aug_scores, no_aug_score.unsqueeze(1)), dim=1)
            aug_scores = torch.cat((aug_scores, aug_score.unsqueeze(1)), dim=1)
            
            # Calculate gaps
            no_aug_gap = [(no_aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
            aug_gap = [(aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
            
            no_aug_gap_list.append(round(sum(no_aug_gap) / len(no_aug_gap), 4))
            aug_gap_list.append(round(sum(aug_gap) / len(aug_gap), 4))
        
        # Move to CPU for final processing if needed
        no_aug_scores = no_aug_scores.cpu()
        aug_scores = aug_scores.cpu()
        
        # Calculate population-level (Mixture of Experts) performance
        moe_no_aug_score = no_aug_scores.min(1)[0]
        moe_aug_score = aug_scores.min(1)[0]
        
        moe_no_aug_gap = [(moe_no_aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
        moe_aug_gap = [(moe_aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
        
        moe_no_aug_gap_avg = sum(moe_no_aug_gap) / len(moe_no_aug_gap)
        moe_aug_gap_avg = sum(moe_aug_gap) / len(moe_aug_gap)
        
        moe_no_aug_score_avg = moe_no_aug_score.mean().item()
        moe_aug_score_avg = moe_aug_score.mean().item()
        
        # Log results
        self.logger.info(f">> Val Score on {val_path}: NO_AUG_Score: {no_aug_score_list} -> "
                        f"Min {min(no_aug_score_list)} Col {moe_no_aug_score_avg}, "
                        f"AUG_Score: {aug_score_list} -> Min {min(aug_score_list)} -> Col {moe_aug_score_avg}")
        
        self.logger.info(f">> Val Score on {val_path}: NO_AUG_Gap: {no_aug_gap_list} -> "
                        f"Min {min(no_aug_gap_list)}% -> Col {moe_no_aug_gap_avg}%, "
                        f"AUG_Gap: {aug_gap_list} -> Min {min(aug_gap_list)}% -> Col {moe_aug_gap_avg}%")
        
        end_time = time.time()
        inference_time = end_time - start_time
        self.logger.info(f">> Inference Time on {val_path}: {inference_time:.2f} seconds")
        
        # Save inference time to result_log
        if hasattr(self, 'current_epoch'):
            epoch = self.current_epoch
        else:
            epoch = 0  # fallback
        
        self.result_log.append(f'inference_time_{val_path.replace(".pkl", "")}', epoch, inference_time)
        return moe_aug_score_avg, moe_aug_gap_avg
