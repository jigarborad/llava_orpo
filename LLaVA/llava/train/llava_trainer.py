import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
import torch.nn.functional as F
from typing import Dict, Union, Any, Tuple, List, Optional
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.modeling_utils import unwrap_model, WEIGHTS_NAME
from transformers.utils import SAFE_WEIGHTS_NAME
from safetensors.torch import save_file as safe_save_file
from transformers import TrainerState, TrainerControl
import numpy as np
import json
from transformers import Trainer
from transformers.trainer import (
    PREFIX_CHECKPOINT_DIR,
    is_sagemaker_mp_enabled,
    logger,
)
from transformers.trainer_utils import (
    ShardedDDPOption,
)
from transformers.utils import SAFE_WEIGHTS_NAME, is_torch_tpu_available
try:
    from safetensors.torch import save_file as safe_save_file
except ImportError:
    safe_save_file = None


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    """def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)"""
class LLaVaORPOTrainer(LLaVATrainer):
    def compute_loss(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs=False):
        data_dict = inputs
        win_input_ids = data_dict.pop('win_input_ids')
        rej_input_ids = data_dict.pop('rej_input_ids')
        win_labels = data_dict.pop('win_labels')
        rej_labels = data_dict.pop('rej_labels')
        images = data_dict.pop('images')

        concatenated_input_ids = data_dict.pop('concatenated_input_ids')
        concatenated_labels = data_dict.pop('concatenated_labels')
        concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')
        concatenated_images = torch.cat([images, images], dim=0)

        concatenated_logits = self.forward_pass(model,
                                                concatenated_input_ids,
                                                concatenated_labels,
                                                concatenated_attention_mask,
                                                concatenated_images,
                                                **data_dict)
        
        win_size = win_input_ids.shape[0]
        rej_size = rej_input_ids.shape[0]
        assert win_size == rej_size

        policy_win_logits, policy_rej_logits = concatenated_logits.split([win_size, rej_size])

        policy_win_logps = self.get_batch_logps(policy_win_logits, win_labels)
        policy_rej_logps = self.get_batch_logps(policy_rej_logits, rej_labels)

        losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = self.odds_ratio_loss(
            policy_win_logps, policy_rej_logps
        )

        nll_loss = self.compute_nll_loss(concatenated_logits[:win_labels.shape[0]], win_labels[:win_labels.shape[0]])
        loss = nll_loss - losses.mean()
        
        train_test = 'train' if model.training else 'eval'
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        metrics = {
            f'rewards_{train_test}/chosen': chosen_rewards.mean().item(),
            f'rewards_{train_test}/rejected': rejected_rewards.mean().item(),
            f'rewards_{train_test}/accuracies': reward_accuracies.mean().item(),
            f'rewards_{train_test}/margins': (chosen_rewards - rejected_rewards).mean().item(),
            f'logps_{train_test}/rejected': policy_rej_logps.mean().item(),
            f'logps_{train_test}/chosen': policy_win_logps.mean().item(),
            f'log_odds_{train_test}/ratio': log_odds_ratio,
            f'log_odds_{train_test}/chosen': log_odds_chosen,
            f'loss_{train_test}/nll': nll_loss,
        }
        #metrics = self.compute_metrics(policy_win_logps, policy_rej_logps, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen, nll_loss.item(), train_test)
        self.log(metrics)

        return loss

    def forward_pass(self, model, input_ids, labels, attention_mask, images, **kwargs):
        output = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            images=images,
            **kwargs
        )
        return output.logits 

    def get_batch_logps(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        labels[labels == -100] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)

    def odds_ratio_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, float, float]:
        log_odds = (policy_chosen_logps - policy_rejected_logps) - (
            torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps))
        )
        sig_ratio = F.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)
        losses = self.args.orpo_beta * ratio

        chosen_rewards = self.args.orpo_beta * policy_chosen_logps.detach()
        rejected_rewards = self.args.orpo_beta * policy_rejected_logps.detach()

        return losses, chosen_rewards, rejected_rewards, torch.mean(ratio).item(), torch.mean(log_odds).item()

    def compute_nll_loss(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        seq_length = labels.size(1)
        logits = logits[:, -seq_length:, :].contiguous()
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Flatten the tensors
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_labels = shift_labels.view(-1)
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(flat_shift_logits, flat_shift_labels)
        
    """def _save_checkpoint(self, model, trial, metrics=None):
        # Convert any non-leaf tensors in the trainer state to detached tensors
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        for key, value in self.state.__dict__.items():
            if isinstance(value, torch.Tensor) and not value.is_leaf:
                setattr(self.state, key, value.detach().clone())
        
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if self.args.should_save:
            self._save(output_dir)
        
        if self.args.should_save:
            self._save_checkpoint_for_llava(model, trial, metrics, output_dir)

    def _save_checkpoint_for_llava(self, model, trial, metrics, output_dir):
        # Save model configuration
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        from transformers.modeling_utils import unwrap_model
        unwrapped_model = unwrap_model(model)
        unwrapped_model.config.save_pretrained(output_dir)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save training arguments
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        
        # Only save Adapter if specified
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])
            
            weight_to_save = get_mm_adapter_state_maybe_zero_3(unwrapped_model.named_parameters(), keys_to_match)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                torch.save(weight_to_save, os.path.join(output_dir, 'mm_projector.bin'))
                
                # Save adapter configuration
                if hasattr(unwrapped_model, 'get_adapter_config'):
                    adapter_config = unwrapped_model.get_adapter_config()
                    adapter_config.save_pretrained(output_dir)
        else:
            # Save the full model
            state_dict = unwrapped_model.state_dict()
            self._save_state_dict(state_dict, output_dir)
        
        # Save README if it doesn't exist
        readme_path = os.path.join(output_dir, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write(f"# LLaVA-ORPO Model\n\nThis is a checkpoint of a LLaVA-ORPO model trained with the following arguments:\n\n```\n{self.args}\n```")

    def _save_state_dict(self, state_dict, output_dir):
        if self.args.save_safetensors and safe_save_file is not None:
            safe_save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
        else:
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))"""
    def _save_checkpoint(self, model, trial, metrics=None):
        # Determine the checkpoint directory
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        
        self.save_model_and_state(output_dir, metrics)
        
        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Maybe delete some older checkpoints.
        if self.is_world_process_zero():
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def save_model_and_state(self, output_dir: str, metrics: Dict[str, float] = None):
        os.makedirs(output_dir, exist_ok=True)

        # Save README.md
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write("LLaVA-ORPO Model Checkpoint\n\nThis checkpoint contains the model state and training information.")

        # Save config files
        self.model.config.save_pretrained(output_dir)

        # Save tokenizer files
        self.tokenizer.save_pretrained(output_dir)

        # Save model weights
        if safe_save_file is not None:
            state_dict = self.model.state_dict()
            state_dict = {k: v.cpu().contiguous() for k, v in state_dict.items()}

            # Check total size
            total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
            max_size = 5 * 1024 * 1024 * 1024  # 5GB

            if total_size > max_size:
                # If larger than 5GB, split into multiple files
                current_size = 0
                current_dict = {}
                file_index = 0

                for k, v in state_dict.items():
                    v_size = v.numel() * v.element_size()
                    if current_size + v_size > max_size:
                        # Save current dict and start a new one
                        safe_save_file(current_dict, os.path.join(output_dir, f"model-{file_index}.safetensors"))
                        current_dict = {k: v}
                        current_size = v_size
                        file_index += 1
                    else:
                        current_dict[k] = v
                        current_size += v_size

                # Save the last part
                if current_dict:
                    safe_save_file(current_dict, os.path.join(output_dir, f"model-{file_index}.safetensors"))

                # Create index file
                index = {"metadata": {}, "weight_map": {k: f"model-{i}.safetensors" for i, k in enumerate(state_dict.keys())}}
                with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
                    json.dump(index, f, indent=2)
            else:
                # If smaller than 5GB, save as a single file
                safe_save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

        # Save training args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save trainer state
        if metrics is not None:
            self.state.log_history.append(metrics)
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override the save_model method to use our custom saving logic
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self.save_model_and_state(output_dir)