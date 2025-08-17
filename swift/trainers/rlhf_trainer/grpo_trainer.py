# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/trl.
import concurrent.futures
import inspect
import os
import re
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field
from math import ceil
from queue import Queue
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import datetime

import datasets
import torch
import torch.nn as nn
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from packaging import version
from torch.nn import ModuleList
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TrainerCallback
from transformers.trainer import Trainer
from transformers.trainer_utils import seed_worker
from trl import GRPOTrainer as HFGRPOTrainer
from trl.extras.profiling import profiling_decorator
from trl.models import prepare_deepspeed
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_trainer import nanmax, nanmin

from swift.llm import InferRequest, MultiModelKeys, RequestConfig, RowPreprocessor, get_model_arch, to_device
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.plugin import loss_scale_map, multi_turns, orms, rm_plugins
from swift.utils import JsonlWriter, gc_collect, get_device, get_logger, is_vllm_available, is_wandb_available
from ..mixin import SwiftMixin
from .rlhf_mixin import RLHFTrainerMixin
from .utils import patch_lora_merge, patch_lora_unmerge, unwrap_model_for_generation
from .vllm_client import VLLMClient
from swift.trainers.sandbox import execute_code_in_sandbox
del HFGRPOTrainer.__init__
del HFGRPOTrainer.log
import json
import random
import string
import pickle
import timeout_decorator    # For more efficient timeout control
logger = get_logger()
if is_wandb_available():
    import wandb

InputsType = List[Dict[str, Union[torch.Tensor, Any]]]
# tuple: (messages, finish_reason)
OutputsType = List[Tuple[List[Dict], str]]
from PIL import Image
INVALID_REWARD_VALUE = -100
def check_aspect_ratio(img_path):
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            aspect_ratio = max(width, height) / min(width, height)
            return aspect_ratio <= 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def _remove_unpickable_values(dictionary):
    if dictionary is None:
        return None
    def is_pickable(obj):
        try:
            pickle.dumps(obj)
            return True
        except Exception:
            return False

    keys_to_remove = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            _remove_unpickable_values(value)
        elif not is_pickable(value):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del dictionary[key]
    return dictionary

def generate_unique_string(existing_strings, length=6):
    # 生成一个唯一的字符串
    while True:
        # 生成一个随机的六位字符串
        new_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        # 检查是否已经存在
        if new_string not in existing_strings:
            existing_strings.add(new_string)
            return new_string

def get_filename_without_extension(file_path):
    # 获取文件名（带后缀）
    file_name_with_extension = os.path.basename(file_path)
    # 分离文件名和后缀
    file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
    return file_name_without_extension


def has_repeated_content(s: str, threshold: float = 0.3, tail_length: int = 1000) -> bool:
    """
    仅检测字符串末尾指定长度（默认为500）的部分是否存在重复内容。
    这使得检测时间对于长字符串来说是恒定的。

    :param s: 输入字符串
    :param threshold: 重复内容的比例阈值
    :param tail_length: 要分析的字符串末尾的长度
    :return: (是否超过阈值, 替换后的字符串)
    """
    if 'sandbox_output' in s or 'answer' in s or '<code>' in s:
        return False

    log_dir = 'logs/reward_log'
    log_file_path = os.path.join(log_dir, 'rl_max3072_output_zhiweikong_gene8_code_orm.txt')
    # 如果字符串本身就比要分析的长度短，那就分析整个字符串
    if len(s) <= tail_length:
        target_string = s
    else:
        # 否则，只取最后 `tail_length` 个字符进行分析
        target_string = s[-tail_length:]

    # 调用核心函数进行检测
    result = has_repeated_content_optimized(target_string, threshold)

    # 如果检测到了重复内容
    if result:
        repeated_substr, count, repeated_ratio = result

        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 准备写入文件的日志信息
        log_entry = (
            f"--- [Repeated Content Detected] ---\n"
            f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Threshold: {threshold:.2f}\n"
            f"Detected Ratio: {repeated_ratio:.2f}\n"
            f"Repeated Substring: \"{repeated_substr}\"\n"
            f"Count: {count}\n"
            f"Analyzed Tail ({len(target_string)} chars): \"{target_string}\"\n"
            f"Full Original String ({len(s)} chars): \"{s}\"\n"
            f"-------------------------------------\n\n"
        )
        
        # 将日志信息追加到文件中
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # 返回True表示检测成功
        return True

    # 如果没有检测到重复内容，返回False
    return False


def has_repeated_content_optimized(s: str, threshold: float = 0.5) -> Optional[Tuple[str, int, float]]:
    """
    在字符串中检测重复内容。

    :param s: 输入字符串
    :param threshold: 重复内容的比例阈值
    :return: 如果找到，则返回一个包含(重复子串, 次数, 重复比例)的元组，否则返回 None
    """
    n = len(s)
    # 对于短字符串，不进行检测
    if n < 50:  
        return None

    # 从长到短遍历所有可能的子串长度
    for length in range(n // 3, 0, -1):
        # 优化：如果一个子串即使重复两次也无法超过阈值，那么更短的子串更不可能
        if (2 * length) / n <= threshold:
            break

        counts = {}
        # 统计当前长度的所有子串的出现次数
        for i in range(n - length + 1):
            current_substr = s[i:i + length]
            counts[current_substr] = counts.get(current_substr, 0) + 1
        
        # 检查是否有子串的重复比例超过阈值
        for substr, count in counts.items():
            if count > 1:
                repeated_ratio = (count * length) / n
                if repeated_ratio > threshold:
                    # 找到了，返回详细信息
                    return substr, count, repeated_ratio
    
    # 遍历结束，没有找到符合条件的重复内容
    return None

class GRPOCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer

    # offload original_modules to cpu, to save memory
    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer.queue = self.trainer.train_queue
        train_dataloader = getattr(state, 'train_dataloader', None) or kwargs.get('train_dataloader')
        self.trainer._prefetch(train_dataloader)


@dataclass
class DataCache:
    inputs: List[Dict] = field(default_factory=list)
    outputs: List[Dict] = field(default_factory=list)

REASONING_SYS_PROMPT='''You are a helpful assistant.

Solve the following problem step by step, and optionally write Python code for image manipulation to enhance your reasoning process. The Python code will be executed by an external sandbox, and the processed image or result (wrapped in <sandbox_output></sandbox_output>) can be returned to aid your reasoning and help you arrive at the final answer.

**Reasoning & Image Manipulation (Optional but Encouraged):**
    * You have the capability to write executable Python code to perform image manipulations (e.g., cropping to a Region of Interest (ROI), resizing, rotation, adjusting contrast) or perform calculation for better reasoning.
    * The code will be executed in a secure sandbox, and its output will be provided back to you for further analysis.
    * All Python code snippets **must** be wrapped as follows:
    <code>
    ```python
    # your code.
    ```
    </code>
    * At the end of the code, print the path of the processed image (processed_path) or the result for further processing in a sandbox environment.'''


class GRPOTrainer(RLHFTrainerMixin, SwiftMixin, HFGRPOTrainer):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __init__(self,
                 model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 reward_model: Optional[List[Union[PreTrainedModel, nn.Module]]] = None,
                 reward_funcs: Optional[List[Union[str, Callable]]] = None,
                 *_args,
                 **kwargs):
        from swift.trainers.rlhf_arguments import GRPOConfig
        args: GRPOConfig = kwargs['args']
        self.existing_strings = set()
        self.args = args
        self.O3 = args.O3
        self.start_time = str(os.environ.get("TIME_STAMP", "0624_14:35:58"))
        logger.warning(f"start_time: {self.start_time}")
        if self.O3:
            logger.warning("You are using O3 paradigm.")
        # for async generate
        self.train_queue = Queue()
        self.eval_queue = Queue()

        self.processing_class = kwargs.get('template').tokenizer

        # for offload model/optimizer
        self.offload_modules = {}
        self.offload_states = {}

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        if reward_funcs:
            for i, reward_func in enumerate(reward_funcs):
                if reward_func in orms:
                    reward_func_class = orms[reward_func]
                    reward_func_args = list(inspect.signature(reward_func_class.__init__).parameters)
                    reward_func_kwargs = {
                        key: getattr(args, key)
                        for key in reward_func_args if key not in ['self', 'args', 'kwargs'] and hasattr(args, key)
                    }
                    if 'tokenizer' in reward_func_args:
                        reward_func_kwargs['tokenizer'] = self.processing_class
                    reward_funcs[i] = reward_func_class(**reward_func_kwargs)
                elif not callable(reward_func):
                    raise ValueError(f'reward_function {reward_func} is not implemented in swift.llm.plugin')

        self.reward_funcs = reward_funcs
        self.reward_func_names = []
        for reward_func in reward_funcs:
            if inspect.isfunction(reward_func):
                reward_func_name = reward_func.__name__
            else:
                reward_func_name = reward_func.__class__.__name__
            self.reward_func_names.append(reward_func_name)

        self.reward_model_plugins = [None] * len(self.reward_funcs)

        if reward_model is not None:
            reward_template = kwargs.pop('reward_template')
            reward_plugins = args.reward_model_plugin
            if reward_plugins is None:
                reward_plugins = ['default'] * len(reward_model)
            assert len(reward_plugins) == len(reward_model), (
                f"The number of 'reward_model_plugin' ({len(reward_plugins)}) does not match "
                f"the number of 'reward_model' ({len(reward_model)}). "
                "Please provide a corresponding 'reward_model_plugin' for each 'reward_model'.")
            for rm, rm_plugin, rm_template in zip(reward_model, reward_plugins, reward_template):
                # Set encoding mode train(see details in Template.encode).
                # Set max_length to None to disable truncation, as the input length has already been truncated earlier.
                rm_template.set_mode('train')
                rm_template.max_length = None
                if rm_plugin not in rm_plugins:
                    raise ValueError(f'rm_plugin {rm_plugin} is not implemented in swift.llm.plugin')
                self.reward_model_plugins.append(rm_plugins[rm_plugin](model=rm, template=rm_template))
                self.reward_funcs.append(rm)
                self.reward_func_names.append(rm.config._name_or_path.split('/')[-1])

        if not self.reward_funcs:
            raise ValueError('You must specify reward_funcs or reward_model')

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(f'Number of reward weights ({len(args.reward_weights)}) must match number of reward '
                                 f'functions ({len(reward_funcs)})')
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        self.multi_turn_func = None
        if self.args.multi_turn_func:
            if isinstance(self.args.multi_turn_func, str):
                assert self.args.multi_turn_func in multi_turns
                multi_turn_func = multi_turns[self.args.multi_turn_func]
                self.multi_turn_func = multi_turn_func
            else:
                self.multi_turn_func = self.args.multi_turn_func

        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.loss_type = args.loss_type
        self.max_completion_length = args.max_completion_length
        self.completion_length_limit_scope = args.completion_length_limit_scope
        model.warnings_issued['estimate_tokens'] = True
        kwargs['data_collator'] = lambda features: features
        self.shuffle_dataset = args.dataset_shuffle

        self.use_vllm = args.use_vllm
        self.async_generate = args.async_generate
        vllm_client = kwargs.pop('vllm_client')  # for external vllm

        super().__init__(model, ref_model, *_args, **kwargs)

        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            'prompt': deque(maxlen=maxlen),
            'completion': deque(maxlen=maxlen),
            'rewards': defaultdict(lambda: deque(maxlen=maxlen)),
        }

        # num_generation check
        num_processes = self.accelerator.num_processes
        self.effective_train_batch_size = effective_batch_size = \
            args.per_device_train_batch_size * num_processes * args.gradient_accumulation_steps
        possible_values = [n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0]

        if self.num_generations not in possible_values:
            raise ValueError(
                f'The effective train batch size ({num_processes} x {args.per_device_train_batch_size} x '
                f'{args.gradient_accumulation_steps}) must be evenly divisible by the number of generations per '
                f'prompt ({self.num_generations}). Given the current effective train batch size, the valid values for '
                f'the number of generations are: {possible_values}.')
        if self.args.eval_strategy != 'no':
            effective_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f'The effective eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be '
                    f'evenly divisible by the number of generations per prompt ({self.num_generations}). Given the '
                    'current effective eval batch size, the valid values for the number of generations are: '
                    f'{possible_values}.')

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        self.parameter_groups, self.parameter_groups_no_lora = self.split_batches()
        self.use_fast_infer = self.use_vllm  # whether to use the PT backend
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError('vLLM is not available and `use_vllm` is set to True. '
                                  'Please install vLLM with `pip install vllm -U` to use it.')
            if self.vllm_mode == 'server':
                self.vllm_client: VLLMClient = vllm_client
            elif self.vllm_mode == 'colocate':
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size '
                        f'({self.accelerator.num_processes}) evenly.')

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration([
                        list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                        for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                    ])

                self.engine = self.prepare_vllm(model)
                # Avoid thread-unsafe modifications of the mode.
                self.engine.default_template = copy(self.template)  # Avoid thread-unsafe modifications of the mode.
        else:
            from swift.llm import PtEngine
            self.engine = PtEngine.from_model_template(self.model, copy(self.template), max_batch_size=0)  # 0: no limit

        self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation
        self.request_config = RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words,
        )
        print("-"* 15 + "Stop Words" + "-"* 15)
        print(self.request_config.stop)
        print("-"* 40)
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True)

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle. # noqa
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        if self.async_generate:
            self.add_callback(GRPOCallback(self))

        if self.args.dynamic_sample:
            self.resample_dataset = deepcopy(self.train_dataset)

            def cyclic_iter(iterable):
                while True:
                    for x in iterable:
                        yield x

            self.resample_iterator = cyclic_iter(self.get_resample_dataloader())
        # flag indicating whether the evaluation has started
        self.eval_flag = False
        self.gen_iter_limit = int(os.getenv('GEN_ITER_LIMIT', 4))
        print(f"Current ITERATION LIMIT: {self.gen_iter_limit}")

    @profiling_decorator
    def _prepare_inputs(
            self, accumulated_local_batch: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = 'train' if self.model.training else 'eval'
        if mode == 'train':
            generate_every = self.args.gradient_accumulation_steps * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                accumulated_local_batch = self._generate_and_score_completions(accumulated_local_batch)
                self._buffered_inputs = accumulated_local_batch  # < this is the change
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(accumulated_local_batch)
        return inputs

    def split_batches(self):
        """Sync weights in batches
        Only split LLM layers for now:
        1. N batches for layers
        2. other, embeds, lm_heads in one batch
        3. multi-modal components in one batch
        """
        model = self.accelerator.unwrap_model(self.model)
        if self.args.move_model_batches is None:
            # All in one
            return [[n for n, p in model.named_parameters() if 'ref_model' not in n]], [None]

        model_arch = get_model_arch(model.model_meta.model_arch)
        non_llm_parameters = []
        llm_embeds = []
        parameters = []
        pattern = r'\.(\d+)\.'

        layer_count = None
        # Get the number of layers in LLM modules
        for name, module in model.named_modules():
            if isinstance(module, ModuleList):
                if model_arch is not None and isinstance(model_arch, MultiModelKeys):
                    llm = model_arch.language_model
                    vision_tower = model_arch.vision_tower
                    if any(vt in name for vt in vision_tower):
                        continue
                    if isinstance(llm, list):
                        llm = llm[0]
                    if name.startswith('base_model'):
                        name = name.replace('base_model.', '')
                    if llm in name:
                        layer_count = len(module)
                else:
                    layer_count = len(module)
        assert layer_count is not None, 'Cannot find ModuleList to split modules.'

        n_layers = ceil(layer_count / self.args.move_model_batches)
        for _ in range(self.args.move_model_batches):
            parameters.append([])

        def replace_lora(name):
            if 'lora_' in name:
                return ''
            else:
                return name.replace('base_layer.', '')

        def remove_lora_and_prefix(names):
            names = set([re.sub(r'^_model\.', '', replace_lora(n)) for n in names])
            return [n for n in names if n]

        def split_llm(name):
            match = re.search(pattern, name)
            if match:
                number = match.group(1)
                group = int(number) // n_layers
                parameters[group].append(name)
            else:
                llm_embeds.append(name)

        for name, parameter in model.named_parameters():
            if 'ref_model' in name:
                continue
            if model_arch is not None and isinstance(model_arch, MultiModelKeys):
                llm = model_arch.language_model
                vision_tower = model_arch.vision_tower
                if any(vt in name for vt in vision_tower):
                    non_llm_parameters.append(name)
                elif isinstance(llm, list):
                    llm = llm[0]
                    if llm in name:
                        split_llm(name)
                    else:
                        non_llm_parameters.append(name)
            else:
                split_llm(name)

        if llm_embeds:
            parameters.append(llm_embeds)
        if non_llm_parameters:
            parameters.append(non_llm_parameters)
        parameters = [p for p in parameters if p]
        parameters_no_lora = [remove_lora_and_prefix(p_list) for p_list in parameters]
        return parameters, parameters_no_lora

    def prepare_vllm(self, model):
        from swift.tuners import Swift
        # from swift.llm.infer.infer_engine import GRPOVllmEngine
        from swift.llm.infer.infer_engine import GRPOVllmEngine_ATS
        if self.vllm_tensor_parallel_size > 1:
            vllm_kwargs = {'distributed_executor_backend': 'external_launcher'}
        else:
            vllm_kwargs = {}

        engine_kwargs = {'seed': self.accelerator.process_index // self.vllm_tensor_parallel_size}

        max_num_seqs = (
            self.args.per_device_train_batch_size * self.vllm_tensor_parallel_size
            * self.args.gradient_accumulation_steps)
        current_device = get_device()
        self.template.O3 = self.O3
        with Swift.grpo_context(model, self.template.processor):
            # engine = GRPOVllmEngine(
            engine = GRPOVllmEngine_ATS(
                model.model_dir,
                model.model_info.torch_dtype,
                model_type=model.model_meta.model_type,
                tensor_parallel_size=self.vllm_tensor_parallel_size,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                enable_prefix_caching=self.args.vllm_enable_prefix_caching,
                max_num_seqs=max_num_seqs,
                enforce_eager=self.args.vllm_enforce_eager,
                limit_mm_per_prompt=self.args.vllm_limit_mm_per_prompt,
                enable_sleep_mode=self.args.sleep_level > 0,
                use_async_engine=False,
                device=current_device,
                max_model_len=self.args.vllm_max_model_len,
                engine_kwargs=engine_kwargs,
                **vllm_kwargs)
            engine.default_template = self.template
        return engine

    @contextmanager
    def _template_context(self, template):
        # The max_length for prompt and completion has already been restricted, so there is no need for max_length here.
        max_length = template.max_length
        mode = template.mode
        if mode in {'vllm', 'pt', 'lmdeploy'}:
            template.set_mode('train')
        template.max_length = None
        loss_scale = template.loss_scale
        if self.multi_turn_func:
            template.loss_scale = loss_scale_map['default']()
        try:
            yield
        finally:
            template.loss_scale = loss_scale
            template.set_mode(mode)
            template.max_length = max_length

    @profiling_decorator
    def _move_model_to_vllm(self):
        if self.vllm_mode == 'server':
            return super()._move_model_to_vllm()

        from accelerate.utils.other import is_compiled_module

        for i, parameter_group in enumerate(self.parameter_groups):
            parameter_group_no_lora = self.parameter_groups_no_lora[i]
            with unwrap_model_for_generation(
                    self.model,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    gather_parameters=parameter_group) as unwrapped_model:

                if is_compiled_module(unwrapped_model):
                    unwrapped_model = unwrapped_model._orig_mod
                if is_peft_model(unwrapped_model):
                    with patch_lora_merge(unwrapped_model, parameter_group):
                        unwrapped_model.merge_adapter()
                    state_dict = unwrapped_model.state_dict()
                    # Remove base_model and base_layer prefixes
                    state_dict = {
                        k.removeprefix('base_model.model.').replace('.base_layer', ''): v
                        for k, v in state_dict.items()
                    }
                    # Remove values with adapter prefix (example: "_lora")
                    state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                    # When module to save, remove its prefix and discard the original module
                    state_dict = {
                        k.replace('modules_to_save.default.', ''): v
                        for k, v in state_dict.items() if 'original_module' not in k
                    }
                else:
                    state_dict = unwrapped_model.state_dict()
                if parameter_group_no_lora:
                    parameter_group_no_lora = [n.replace('base_model.model.', '') for n in parameter_group_no_lora]
                    state_dict = {k: v for k, v in state_dict.items() if k in parameter_group_no_lora}
                assert len(state_dict) > 0 and all([state.shape != torch.Size([0]) for state in state_dict.values()])
                if self.use_fast_infer:
                    if self.args.async_generate:
                        # before sync weight, we should wait async generate finish
                        self._wait_queue()
                    if self.args.use_vllm:
                        llm_model = self.engine.inner_model
                    else:
                        llm_model = self.engine.engine.engine
                    llm_model.load_weights(state_dict.items())
                    del state_dict
                    gc_collect()
                # Unmerge the adapter to restore the model to its original state.
                # This must be done after loading weights to ensure they correspond to the merged state.
                if is_peft_model(unwrapped_model):
                    with patch_lora_unmerge(unwrapped_model):
                        unwrapped_model.unmerge_adapter()
        if self.use_vllm and self.vllm_mode == 'colocate':
            # since vLLM model weights has been updated, we should reset the prefix cache
            self.engine.engine.reset_prefix_cache()

    def _wait_queue(self):
        while self._queue.empty():
            time.sleep(0.01)

    def _infer(self,
               inputs: Optional[InputsType],
               request_config: RequestConfig,
               is_global_inputs: bool = False) -> OutputsType:
        from swift.llm.infer.protocol import ChatCompletionResponse
        request_config = copy(request_config)
        # keys from InferRequest
        per_device_size = len(inputs)
        if is_global_inputs:
            per_device_size //= self.accelerator.num_processes
        infer_inputs = [{
            k: v
            for k, v in inp.items() if k in ['messages', 'images', 'audios', 'videos', 'tools', 'objects']
        } for inp in inputs] if inputs else []
        if self.vllm_mode == 'server':
            # for server mode, we gather all the inputs and send to remote vllm server in main process
            if is_global_inputs:
                all_inputs = infer_inputs
                all_input_lengths = [per_device_size] + [0] * (self.accelerator.num_processes - 1)
            else:
                all_inputs = gather_object(infer_inputs)
                all_input_lengths = gather_object([len(infer_inputs)])

            if not any(inputs for inputs in all_inputs):
                return []

            if self.accelerator.is_main_process:
                results: List[ChatCompletionResponse] = self._engine_infer(
                    infer_requests=all_inputs, request_config=request_config)
            else:
                results = [None] * len(all_inputs)
            # Broadcast the results from the main process to all processes,
            # ensuring each process receives its corresponding slice.
            if not is_global_inputs:
                results = broadcast_object_list(results, from_process=0)
                start_idx = sum(all_input_lengths[:self.accelerator.process_index])
                end_idx = start_idx + all_input_lengths[self.accelerator.process_index]
                results = results[start_idx:end_idx]
            else:
                results = results if self.accelerator.is_main_process else []
        else:
            # pt / vllm
            if self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                # Note: The input sizes may differ across ranks (e.g., in multi-turn scenarios,
                # the amount of data each rank continues to process may vary).
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                local_input_length = len(inputs)
                all_input_lengths = [None] * self.vllm_tensor_parallel_size
                torch.distributed.all_gather_object(all_input_lengths, local_input_length, group=self.tp_group)
                start_idx = sum(all_input_lengths[:local_rank_in_group])
                end_idx = start_idx + all_input_lengths[local_rank_in_group]

                # orig_size = len(inputs)/
                gathered_inputs = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_inputs, inputs, group=self.tp_group)
                inputs = [p for sublist in gathered_inputs for p in sublist]
            # Set request_config.seed
            # 1. Ensure that the seed for vLLM Engines within each TP (Tensor Parallelism) group is the same;
            #   otherwise, the program may hang.
            # 2. Ensure that the seed for vLLM Engines across different TP groups is different;
            #   otherwise, identical completions will be generated.
            mode = 'train' if self.model.training else 'eval'
            batch_size = (
                self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps if mode == 'train' else self.args.per_device_eval_batch_size)
            # Since the TP (Tensor Parallelism) group gathers the inputs,
            # multiply the batch size by the TP parallel size.
            batch_size *= self.vllm_tensor_parallel_size
            request_config.seed = batch_size * (self.accelerator.process_index // self.vllm_tensor_parallel_size)

            results: List[ChatCompletionResponse] = self._engine_infer(
                infer_requests=inputs, request_config=request_config)

            if self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                results = results[start_idx:end_idx]

        return results

    def _set_inputs_system(self, inputs: InputsType) -> InputsType:
        if all(_input['messages'][0]['role'] == 'system' for _input in inputs):
            return
        for _input in inputs:
            messages = _input['messages']
            if messages[0]['role'] != 'system':
                messages.insert(0, {'role': 'system', 'content': self.template.template_meta.default_system})

    def _infer_single_or_multi_turn(self,
                                    inputs: InputsType,
                                    request_config: RequestConfig,
                                    is_global_inputs: bool = False) -> OutputsType:
        """Perform multi-turn or single-turn inference

        Args:
            inputs: list of input requests
            request_config: Inference configuration parameters
            is_global_inputs:
                A boolean indicating whether the inputs are global. When set to True,
                the returned results in the main process will be a complete list of
                global_outputs, while other processes will return an empty list [].
        Returns:
            List of outputs where each entry contains:
            - List of responses per prompt
            - Each response is a tuple of (message_history, finish_reason)
        """
        self._set_inputs_system(inputs)
        # infer first turn
        results = self._infer(inputs, request_config, is_global_inputs)

        if not self.multi_turn_func:
            # Single-turn: combine completions with messages and retain the finish reason.
            outputs = []
            for i, output in enumerate(results):
                _choices = []
                for choice in output.choices:
                    _input: Dict = deepcopy(inputs[i])
                    InferRequest.remove_response(_input['messages'])
                    _input['messages'].append({'role': 'assistant', 'content': choice.message.content})
                    _choices.append((_input['messages'], choice.finish_reason))
                outputs.append(_choices)
            # flatten 2D list to 1D list
            outputs = [item for sublist in outputs for item in sublist]
        else:
            # Multi-turn: continue to rollout until finished.
            orig_size = len(inputs)
            outputs = [None] * orig_size
            # we remove origin response in first turn
            first_turn = True
            next_turn_inputs = inputs.copy()
            last_turn_results = results
            while True:
                has_local_data = len(next_turn_inputs) > 0
                has_global_data = gather_object([has_local_data])
                if not any(has_global_data):
                    break
                # inputs for current turn
                current_inputs = []
                cnt = 0
                # combine completions from results with messages
                for i, output in enumerate(last_turn_results):
                    for choice in output.choices:
                        current_input = deepcopy(next_turn_inputs[i])
                        messages = current_input['messages']

                        # Determine whether to append a new message or update the last one based on the current state
                        if first_turn or not messages[-1]['content'] or messages[-1]['content'] == '<None>':
                            # If it's the first turn or the last message content is empty(dummy), remove the response
                            InferRequest.remove_response(messages)
                        if messages[-1]['role'] == 'assistant':
                            # If the last message was assistant, concatenate the new content to it
                            messages[-1]['content'] += choice.message.content
                        else:
                            # append a new message from the assistant
                            messages.append({'role': 'assistant', 'content': choice.message.content})

                        if 'index' not in current_input:
                            current_input['index'] = cnt
                        current_input['finish_reason'] = choice.finish_reason
                        cnt += 1
                        current_inputs.append(current_input)

                # Process messages in the multi-turn function
                current_results: List[Dict] = self.multi_turn_func(current_inputs) if has_local_data else []

                # Retain messages that are not yet finished for the next round of rollout
                pending_inputs = []
                for r in current_results:
                    if r['finished'] or r['finish_reason'] == 'length':
                        outputs[r['index']] = (r['messages'], r['finish_reason'])
                    else:
                        if r['messages'][-1]['role'] == 'assistant':
                            # Sometimes, after processing with multi_turn_func,
                            # we want to continue reasoning based on the previous assistant content.
                            # However, _infer will remove the response internally, so we add a dummy response here
                            # to prevent the assistant content from being removed.
                            r['messages'].append({'role': 'assistant', 'content': '<None>'})
                        pending_inputs.append(r)

                current_infer_inputs = pending_inputs if has_local_data else []
                current_results = self._infer(current_infer_inputs, request_config)

                last_turn_results = current_results
                next_turn_inputs = pending_inputs
                first_turn = False

            assert not any([o is None for o in outputs])
        return outputs

    def async_infer(self, all_inputs):
        current_queue = self._queue

        def infer_task():
            try:
                with self.multi_turn_completion_length_context():
                    return self._infer_single_or_multi_turn(all_inputs, self.request_config, is_global_inputs=True)
            except Exception as e:
                logger.error('Inference task failed: %s', str(e))
                raise

        future: Future = self.executor.submit(infer_task)

        # pre-fetch the queue to avoid switching back to eval_queue at the end of training sample sampling

        def done(future):
            try:
                result = future.result()
                current_queue.put(DataCache(all_inputs, result))
            except Exception as e:
                logger.error('Error in async_infer callback: %s', str(e))

        future.add_done_callback(done)

    def _prefetch(self, dataloader: DataLoader):
        inputs = next(iter(dataloader))
        all_inputs = gather_object(inputs)
        outputs = self._infer_single_or_multi_turn(all_inputs, self.request_config, is_global_inputs=True)
        self._queue.put(DataCache(all_inputs, outputs))


    def process_round(self, gen_inputs, gen_outputs, previous_execution_contexts, cur_round=1000):
        if len(gen_inputs) == 0:
            return gen_inputs, previous_execution_contexts
        cache_dir = f"/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/RL_case_cache_new/{self.start_time}_{self.accelerator.process_index}"
        os.makedirs(cache_dir, exist_ok=True)

        assert len(gen_outputs) == len(gen_inputs)

        for idx in range(len(gen_outputs)):
            # previous_execution_context = previous_execution_contexts[idx]
            if gen_outputs[idx] is not None:
                gen_output = gen_outputs[idx][0][-1]
                assert gen_output["role"] == "assistant", f"Line 937 assert error: {gen_outputs[idx][0]}"
                if gen_inputs[idx]["messages"][-1]["role"] == "assistant":
                    if "Execution timed out after" in gen_inputs[idx]["messages"][-1]["content"]:
                        context = gen_inputs[idx]["messages"][-1]["content"]
                        print(f"Already Timeout: {context}")
                        continue
                    if "</answer>" in gen_inputs[idx]["messages"][-1]["content"]:
                        # context = gen_inputs[idx]["messages"][-1]["content"]
                        # print(f"Already Timeout: {context}")
                        continue
                response = gen_output["content"]
                
                if response.strip() in ["</code>", "</answer>", "<|im_end|>"]:
                    response = ""
                if has_repeated_content(response):
                    response = ""
                
                if ( "images" not in gen_inputs[idx] or
                    not gen_inputs[idx]["images"] or
                    not isinstance(gen_inputs[idx]["images"][0], dict) or
                    "path" not in gen_inputs[idx]["images"][0]):
                    
                    image_path = "/mmu_mllm_hdd_2/yifanzhang/1.jpeg"
                else:
                    image_path = gen_inputs[idx]["images"][0]["path"]
                case_id, case_ext = os.path.splitext(os.path.basename(image_path))
                generate_code_round = False

                if "<code>" in self.request_config.stop and gen_inputs[idx]["messages"][-1]["role"] == "assistant" and  gen_inputs[idx]["messages"][-1]["content"].strip().endswith("<code>"):
                    generate_code_round = True
                # if "images" not in gen_inputs[idx] or not gen_inputs[idx]["images"] or not len(gen_inputs[idx]["images"]) > 10:
                if not len(gen_inputs[idx]["images"]) > 10:
                    if gen_inputs[idx]["messages"][-1]["role"] == "assistant":
                        gen_inputs[idx]["messages"][-1]["content"] += response
                    else:
                        gen_inputs[idx]["messages"].append({"role": "assistant", "content": response})
                else:
                    continue
                # 开始O3的处理流程, 必须在geninputs之后continue
                if "</answer>" in response:
                    continue
                # Case 2: reach code generation.
                # parse current result. Two cases: reach </code> or reach </answer>
                code_regex = re.compile(r'<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>', re.IGNORECASE)
                if generate_code_round:
                    response = "<code>\n" + response
                code_match = code_regex.search(response)


                # execute code and return result.
                if code_match:
                    gen_inputs[idx]["messages"][-1]["content"] += "<sandbox_output>"
                    code_to_execute = code_match.group(1).strip()
                    case_dir = os.path.join(cache_dir, f"{case_id}_{self.accelerator.process_index}_rnd{cur_round}_{idx}")
                    os.makedirs(case_dir, exist_ok=True)
                    
                    # has_valid_images = False
                    processed_img_paths, captured_stdout, error_msg, current_execution_context = execute_code_in_sandbox(
                        code_to_execute, image_path, temp_output_dir=case_dir, item_id=case_id, previous_execution_context=previous_execution_contexts[idx]
                    )
                    # current_execution_context = _remove_unpickable_values()
                    # print(f"\033[31m--- Found Code Block ---\n{code_to_execute}\n-------------------------\033[0m")
                    previous_execution_contexts[idx] = _remove_unpickable_values(current_execution_context)
                    # import pdb; pdb.set_trace()
                    if not processed_img_paths:
                        if error_msg is None:
                            gen_inputs[idx]["messages"][-1]["content"] += f'Sandbox for {case_id}: No output captured.'
                            print("-" * 30)
                            print(f"Corner Case: {processed_img_paths}")
                            print(f"Code: {code_to_execute}")
                            print("-" * 30)
                        else:
                            gen_inputs[idx]["messages"][-1]["content"] += error_msg
                        # continue      
    
                    # has_valid_images = False
                    # generated_content += [
                    #                     {"type": "text", "text": generated_text_segment},
                    #                     {"type": "text", "text": "<sandbox_output>"}
                    #                 ]
                    else:
                        #     import importlib
                        #     importlib.reload(os)
                        # first_path = processed_img_paths[0]
                        first_path = processed_img_paths[0]
                        # gen_inputs[idx]["messages"][-1]["content"] += "<image>"
                        if os.path.exists(first_path):
                            
                            # Iterate through each path in the list
                            valid_aspect_ratio = True
                            for img_path in processed_img_paths:
                                if os.path.exists(img_path):
                                    if not check_aspect_ratio(img_path):
                                        print(f"Invalid image: {img_path}")
                                        valid_aspect_ratio = False
                                        break
                            if valid_aspect_ratio:
                                for img_path in processed_img_paths:
                                    if os.path.exists(img_path):
                                        # if not has_valid_images: # Add text segments only once per sandbox output block
                                        #     has_valid_images = True
                                        if not len(gen_inputs[idx]["images"]) > 10:
                                            gen_inputs[idx]["messages"][-1]["content"] += "<image>"
                                            gen_inputs[idx]["images"].append({"bytes": None, "path": os.path.abspath(img_path)})
                                        else:
                                            break
                            else:
                                gen_inputs[idx]["messages"][-1]["content"] += f"Sandbox for {case_id}: The Generated img has invalid aspect_ratio"

                                    # generated_content.append({"type": "image", "image": img_path})                                
                        else:
                            # generated_content.append({"type": "text", "text": first_path})
                            gen_inputs[idx]["messages"][-1]["content"] += first_path
                    gen_inputs[idx]["messages"][-1]["content"] += "</sandbox_output>"
                else:
                    # wo code. wo </answer>, 那么基本上一直在生成重复内容，直接break这次
                    if "</answer>" not in response and response != "" and not response.strip().endswith("<code>"):
                        response = ''
                        print(f'wo code. wo </answer>: {response}')
        return gen_inputs, previous_execution_contexts


    def _o3_infer(self, inputs: InputsType) -> Tuple[InputsType, OutputsType]:
        # print(f"Input size: {len(inputs)}")
        # process_index = self.infer_rank
        previous_execution_contexts = [{} for i in range(len(inputs))]
        # from swift.llm.infer.protocol import ChatCompletionResponseChoice, ChatMessage, ChatCompletionResponse
        iteration_limit = self.gen_iter_limit
        gen_inputs = deepcopy(inputs)
        # num_prompt_tokens = [0 for i in range(len(inputs))]
        # num_completion_tokens = [0 for i in range(len(inputs))]
        while iteration_limit > 0:
            # if self.accelerator.process_index == 1:
            #     print(f"Start iteration {3 - iteration_limit}.")
            iteration_limit -= 1
            try:
                gen_inputs, gen_outputs = self._fast_infer(gen_inputs)
            except OSError as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    # 添加一个role为assistant的message，content为IndexError
                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers OSError"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print(f"OS Error: {e}")
                with open(f"/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/debug_os_error.jsonl", "a") as fout:
                    fout.write(f"Wrong Case: {e}\n")
                    for i in range(len(gen_inputs)):
                        fout.write(json.dumps(gen_inputs[i]) +"\n")
                        fout.write(json.dumps(gen_outputs[i]) +"\n")
            except torch.OutOfMemoryError as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers OutOfMemoryError"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print("Out of memory!")
                with open(f"/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/debug_oom.jsonl", "a") as fout:
                    fout.write(f"Wrong Case: {e}\n")
                    for i in range(len(gen_inputs)):
                        fout.write(json.dumps(gen_inputs[i]) +"\n")
                        fout.write(json.dumps(gen_outputs[i]) +"\n")
            except ValueError as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    # 添加一个role为assistant的message，content为IndexError
                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers ValueError"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print(f"Value Error: {e}")
                with open(f"/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/debug_value.jsonl", "a") as fout:
                    fout.write(f"Wrong Case: {e}\n")
                    for i in range(len(gen_inputs)):
                        fout.write(json.dumps(gen_inputs[i]) +"\n")
                        fout.write(json.dumps(gen_outputs[i]) +"\n")
            except IndexError as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    # 添加一个role为assistant的message，content为IndexError
                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers IndexError"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print(f"IndexError: {e}")
                with open(f"/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/debug_idx.jsonl", "a") as fout:
                    fout.write(f"Wrong Case: {e}\n")
                    for i in range(len(gen_inputs)):
                        fout.write(json.dumps(gen_inputs[i]) +"\n")
                        fout.write(json.dumps(gen_outputs[i]) +"\n")
            except timeout_decorator.TimeoutError as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    # 添加一个role为assistant的message，content为IndexError
                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers TimeoutError"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print("Out of time!")
                with open(f"./debug_oot.jsonl", "a") as fout:
                    fout.write(f"Wrong Case: {e}\n")
                    for i in range(len(gen_inputs)):
                        fout.write(json.dumps(gen_inputs[i]) +"\n")
                        fout.write(json.dumps(gen_outputs[i]) +"\n")
            except RuntimeError as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    # 添加一个role为assistant的message，content为IndexError
                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers RuntimeError"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print(f"Runtime Error: {e}!")
                with open(f"/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/debug_runtime.jsonl", "a") as fout:
                    fout.write(f"Wrong Case: {e}\n")
                    for i in range(len(gen_inputs)):
                        fout.write(json.dumps(gen_inputs[i]) +"\n")
                        fout.write(json.dumps(gen_outputs[i]) +"\n")
            except Exception as e:
                gen_outputs = [None] * len(gen_inputs)
                for idx in range(len(gen_outputs)):
                    # 提取role为user和system的部分
                    filtered_messages = [msg for msg in gen_inputs[idx]["messages"] if msg["role"] in ["user", "system"]]

                    # 添加一个role为assistant的message，content为IndexError
                    if (gen_inputs[idx]["messages"][-1]["role"] == "assistant" and not ("This instance triggers" in gen_inputs[idx]["messages"][-1]["content"]) and not ("</answer>" in gen_inputs[idx]["messages"][-1]["content"])) or (gen_inputs[idx]["messages"][-1]["role"] != "assistant"):
                        filtered_messages.append({"role": "assistant", "content": "This instance triggers Exception"})

                        # 进行赋值
                        gen_outputs[idx] = (filtered_messages, "stop")
                    else:
                        gen_outputs[idx] = None
                if self.accelerator.process_index == 1:
                    print(f"Exception: {e}")
                with open(f"/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/debug_others.jsonl", "a") as fout:
                    fout.write(f"Wrong Case: {e}\n")
                    for i in range(len(gen_inputs)):
                        fout.write(json.dumps(gen_inputs[i]) +"\n")
                        fout.write(json.dumps(gen_outputs[i]) +"\n")


            gen_inputs, previous_execution_contexts = self.process_round(gen_inputs, gen_outputs, previous_execution_contexts, cur_round=self.gen_iter_limit-iteration_limit)

        for idx in range(len(gen_outputs)):
            gen_outputs[idx] = (gen_inputs[idx]["messages"], "stop")
            if "</answer>" in gen_outputs[idx][0][-1]["content"] :
                gen_outputs[idx][0][-1]["content"] = gen_outputs[idx][0][-1]["content"].split("</answer>")[0] + "</answer>"
            assert gen_inputs[idx]["messages"][-1]["role"] == "assistant", f"Input: {gen_inputs[idx]}\n\nOutput: {gen_outputs[idx]}"
            gen_inputs[idx]["messages"] = gen_inputs[idx]["messages"][:-1]
        # with open(f"/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/debug_{self.accelerator.process_index}.jsonl", "a") as fout:
        #     fout.write(json.dumps({"final input": gen_inputs[-1], "final output": gen_outputs[-1]}) +"\n")
        return gen_inputs, gen_outputs

    def _fast_infer(self, inputs: InputsType) -> Tuple[InputsType, OutputsType]:
        if self.vllm_mode == 'colocate' and self.args.sleep_level > 0:
            if self.args.offload_model:
                self.offload_model()
            if self.args.offload_optimizer:
                self.offload_optimizer()
            if self.args.gc_collect_after_offload:
                gc_collect()
            # Skip the first wake_up to avoid the warning "Executor is not sleeping"
            if self.engine.inner_model_executor.is_sleeping:
                self.engine.engine.wake_up()
        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        if self.async_generate:
            # send this step data to server
            # we gather inputs outside the thread for prevent potential gather deadlock
            all_inputs = gather_object(inputs)
            self.async_infer(all_inputs)
            # cached data from last step
            data_cache = self._queue.get()
            all_inputs = data_cache.inputs
            all_outputs = gather_object(data_cache.outputs)
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )
            inputs = all_inputs[process_slice]
            outputs = all_outputs[process_slice]

        else:
            with self.multi_turn_completion_length_context():
                outputs = self._infer_single_or_multi_turn(inputs, self.request_config)

        if self.vllm_mode == 'colocate' and self.args.sleep_level > 0:
            self.engine.engine.sleep(level=self.args.sleep_level)
            if self.args.gc_collect_after_offload:
                gc_collect()
            if self.args.offload_model:
                self.load_model()
            if self.args.offload_optimizer:
                self.load_optimizer()
        # print(f"inputs size: {len(inputs)}")
        # print(f"outputs size: {len(outputs)}")
        return inputs, outputs

    def _generate_completions(self, inputs: InputsType) -> InputsType:
        """Generate completions for given inputs using either fast inference or standard PyTorch inference.

        Args:
            inputs: List of input examples containing conversation messages.

        Returns:
            Modified inputs with generated completions added to the last message
            and truncation flag set in 'is_truncated' field.
        """
        mode = 'train' if self.model.training else 'eval'
        if self.use_fast_infer:
            # inputs, outputs = self._fast_infer(inputs)
            if not self.O3:
                inputs, outputs = self._fast_infer(inputs)
            else:
                inputs, outputs = self._o3_infer(inputs)
        else:
            # pt infer
            is_multimodal = self.model.model_meta.is_multimodal
            if is_multimodal:
                models = self.template.remove_post_encode_hook()
            with unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ), self.multi_turn_completion_length_context():
                outputs = self._infer_single_or_multi_turn(inputs, self.request_config)
                if mode == 'train':
                    # In training mode, ensure the model is returned to train() mode after inference
                    # This is necessary as pt engines set the model to eval mode during generation
                    self.model.train()
            if is_multimodal:
                self.template.register_post_encode_hook(models)

        for i, output in enumerate(outputs):
            inputs[i]['messages'] = output[0]
            inputs[i]['is_truncated'] = output[1] == 'length'

        return inputs

    def _generate_and_score_completions(self, inputs: InputsType) -> InputsType:

        inputs = self._generate_completions(inputs)
        total_rewards_per_func, total_rewards, completions = self._score_completions(inputs)
        mode = 'train' if self.model.training else 'eval'

        if self.args.dynamic_sample and mode == 'train':
            # dynamic sampling for std=0 groups
            inputs, total_rewards, total_rewards_per_func, completions = \
                self._dynamic_sampling(inputs, total_rewards, total_rewards_per_func, completions)

        # Prepare final outputs with advantages and other required fields
        batch_encoded_inputs = self._prepare_batch_inputs(inputs, total_rewards)
        # Log metrics
        messages = [inputs[i]['messages'][:-1] for i in range(len(inputs))]

        self._log_metrics(batch_encoded_inputs, messages, completions, total_rewards, total_rewards_per_func)

        return batch_encoded_inputs

    def fill_invalid_value(self, reward_value):
        # refer to _prepare_batch_inputs
        grouped_reward_value = reward_value.view(-1, self.num_generations)

        # Create a mask for invalid values 
        invalid_mask = (grouped_reward_value == INVALID_REWARD_VALUE)

        # Calculate the sum and count of valid values for each group
        valid_sum = torch.where(invalid_mask, torch.tensor(0.0), grouped_reward_value).sum(dim=1, keepdim=True)
        valid_count = (~invalid_mask).sum(dim=1, keepdim=True)

        # Avoid division by zero by setting valid_mean to 0 where valid_count is 0
        valid_mean = torch.where(valid_count == 0, torch.tensor(0.0), valid_sum / valid_count)

        # Replace invalid values with the mean of valid values
        grouped_reward_value = torch.where(invalid_mask, valid_mean, grouped_reward_value)

        return grouped_reward_value.view(-1), bool(invalid_mask.sum() > 0)

    def _score_completions(self, inputs: InputsType) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Score completions using all reward functions

        Args:
            inputs: List of input examples, each containing a 'messages' list with conversation history

        Returns:
            Tuple containing:
            - rewards_per_func: Tensor of shape (num_examples, num_reward_funcs) with individual rewards
            - total_rewards: Tensor of shape (num_examples,) with weighted sum of rewards
            - completions: List of generated completion strings
        """
        device = self.accelerator.device
        completions = [example['messages'][-1]['content'] for example in inputs]
        rewards_per_func = torch.zeros((len(inputs), len(self.reward_funcs)), device=device)
        reward_name_list = []
        for i, (reward_func, reward_model_plugin) in enumerate(zip(self.reward_funcs, self.reward_model_plugins)):
            # reward model
            reward_name = str(reward_func)
            reward_name_list.append(reward_name.split(" object at")[0].replace("<agent_rm.", "").strip())
            if isinstance(reward_func, nn.Module):
                rewards_per_func[:, i] = reward_model_plugin(inputs=inputs)
            # reward function
            else:
                # Repeat all input columns (but "messages" and "completion") to match the number of generations
                reward_kwargs = RowPreprocessor.rows_to_batched(inputs)
                output_reward_func = reward_func(completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        total_rewards_per_func = gather(rewards_per_func)
        total_rewards_per_func_before = total_rewards_per_func.clone()

        # 获取vqa_orm对应的索引
        # print(f"Reward functions: {reward_name_list}")
        # cst_orm_index = reward_name_list.index("CSTORM")
        # vqa_orm_index = self.reward_funcs.index('vqa_orm')
        # mask = total_rewards_per_func[:, cst_orm_index] <= 0
        # for depend_reward in ["VQAORM"]:
        #     # depend_reward_func = orms[depend_reward]
        #     if depend_reward in reward_name_list:
        #         depend_reward_index = reward_name_list.index(depend_reward)
        #         total_rewards_per_func[mask, depend_reward_index] = 0
        vqa_orm_index = reward_name_list.index("VQAORM")
        # vqa_orm_index = self.reward_funcs.index('vqa_orm')
        mask = total_rewards_per_func[:, vqa_orm_index] <= 0
        for depend_reward in ["CODEORM", "CODEPRM", "VQAPRM", "CSTORM", "CODEORMLLM"]:
            # depend_reward_func = orms[depend_reward]
            if depend_reward in reward_name_list:
                depend_reward_index = reward_name_list.index(depend_reward)
                total_rewards_per_func[mask, depend_reward_index] = 0

        cst_orm_index = reward_name_list.index("CSTORM")
        # vqa_orm_index = self.reward_funcs.index('vqa_orm')
        mask = total_rewards_per_func[:, vqa_orm_index] <= 0
        for depend_reward in ["CODEORM", "CODEPRM", "VQAPRM", "CODEORMLLM"]:
            # depend_reward_func = orms[depend_reward]
            if depend_reward in reward_name_list:
                depend_reward_index = reward_name_list.index(depend_reward)
                total_rewards_per_func[mask, depend_reward_index] = 0
        # 创建一个字典来存储所有变量
        data_to_save = {
            'total_rewards_per_func_before': total_rewards_per_func_before,
            'total_rewards_per_func_after': total_rewards_per_func,
            'mask': mask,
            'vqa_orm_index': vqa_orm_index,
            'depend_reward_indices': {depend_reward: self.reward_funcs.index(depend_reward) for depend_reward in ["code_orm", "code_prm", "vqa_prm", "cst_orm"] if depend_reward in self.reward_funcs}
        }
        if (total_rewards_per_func_before - total_rewards_per_func).sum() != 0:
            torch.save(data_to_save, '/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/scripts/rl/add_mask.pth')

        total_rewards = (total_rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        return total_rewards_per_func, total_rewards, completions

    def _dynamic_sampling(self, inputs, rewards, rewards_per_func, completions):
        # DAPO https://arxiv.org/abs/2503.14476
        # Replaces samples with zero-reward-variance groups (std=0)
        resample_count = 0
        valid_samples = []
        valid_rewards = []
        valid_rewards_per_func = []
        valid_completions = []

        origin_data = (inputs, rewards, rewards_per_func, completions)

        while resample_count < self.args.max_resample_times:
            grouped_rewards = rewards.view(-1, self.num_generations)
            group_std = grouped_rewards.std(dim=1)

            valid_mask = (group_std > 0).repeat_interleave(self.num_generations)
            all_inputs = gather_object(inputs)
            valid_samples.extend([inp for inp, mask in zip(all_inputs, valid_mask) if mask])
            valid_rewards.append(rewards[valid_mask])
            valid_rewards_per_func.append(rewards_per_func[valid_mask])
            valid_completions.extend(
                [inp['messages'][-1]['content'] for inp, mask in zip(all_inputs, valid_mask) if mask])

            if len(valid_samples) >= self.effective_train_batch_size:
                break

            inputs = next(self.resample_iterator)
            inputs = Trainer._prepare_inputs(self, inputs)
            inputs = self._generate_completions(inputs)
            rewards_per_func, rewards, completions = self._score_completions(inputs)
            resample_count += 1

        if len(valid_samples) >= self.effective_train_batch_size:
            process_slice = slice(
                self.accelerator.process_index * len(inputs),
                (self.accelerator.process_index + 1) * len(inputs),
            )
            inputs = valid_samples[:self.effective_train_batch_size][process_slice]
            rewards = torch.cat(valid_rewards)[:self.effective_train_batch_size]
            rewards_per_func = torch.cat(valid_rewards_per_func)[:self.effective_train_batch_size]
            completions = valid_completions[:self.effective_train_batch_size][process_slice]
        else:
            logger.warning(f'There are still std=0 groups present after {self.args.max_resample_times} retries.')
            inputs, rewards, rewards_per_func, completions = origin_data

        return inputs, rewards, rewards_per_func, completions

    def _prepare_batch_inputs(self, inputs: InputsType, rewards: torch.Tensor) -> List[InputsType]:
        """
        Prepare the final batch inputs with advantages, ref/old_policy logps and other fields for RL training.

        Args:
            inputs (InputsType): List of input samples. Original shape is [gas*bs] where:
                - gas: gradient accumulation steps
                - bs: per-device batch size
            rewards (torch.Tensor): Tensor of rewards corresponding to the inputs.
                Shape should match the total number of samples (gas*bs*num_generations)

        Returns:
            List[InputsType]: A list of prepared batch inputs, organized as [gas][bs]
        """
        # Compute advantages
        grouped_rewards = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = grouped_rewards.mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = grouped_rewards.std(dim=1).repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards)
        if self.args.scale_rewards:
            advantages /= (std_grouped_rewards + 1e-4)

        adv_mean = advantages.mean()
        adv_std = advantages.std()

        # 使用全局统计数据对 advantages 进行标准化
        # advantages = (advantages - adv_mean) / (adv_std + 1e-8)  # 1e-8 防止除以零

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(inputs),
            (self.accelerator.process_index + 1) * len(inputs),
        )
        advantages = advantages[process_slice]

        mode = 'train' if self.model.training else 'eval'
        bs = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size
        gas = self.args.gradient_accumulation_steps if mode == 'train' else 1

        assert len(inputs) == bs * gas, f'Expected {bs * gas} inputs, got {len(inputs)}'
        gas_chunks = [inputs[i * bs:(i + 1) * bs] for i in range(gas)]

        ga_batch_encoded_inputs = []
        template = self.template

        # Split advantages by GAS chunks
        advantage_chunks = torch.chunk(advantages, gas)

        for i, (batch, batch_advantages) in enumerate(zip(gas_chunks, advantage_chunks)):
            # Encode and process each batch (size=bs)
            with self._template_context(template):
                batch_encoded_inputs = [template.encode(infer_request) for infer_request in batch]
                batch_encoded_inputs = to_device(template.data_collator(batch_encoded_inputs), self.model.device)

            # Process labels and masks
            labels = batch_encoded_inputs.pop('labels')
            logits_to_keep = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
            batch_encoded_inputs.update({
                'completion_mask':
                labels[:, -logits_to_keep:] != -100,
                'truncated_mask':
                torch.tensor([b['is_truncated'] for b in batch], dtype=torch.bool),
                'logits_to_keep':
                logits_to_keep,
                'advantages':
                batch_advantages
            })

            with torch.no_grad():
                batch_encoded_inputs['old_per_token_logps'] = (
                    self._get_per_token_logps(self.model, batch_encoded_inputs) if self.old_policy else None)

                if self.beta == 0.0:
                    ref_per_token_logps = None
                elif self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, batch_encoded_inputs)
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(self.model, batch_encoded_inputs)
                batch_encoded_inputs['ref_per_token_logps'] = ref_per_token_logps

            ga_batch_encoded_inputs.append(batch_encoded_inputs)

        return ga_batch_encoded_inputs

    def _log_metrics(self, inputs, messages, completions, rewards, rewards_per_func):
        """Log training/evaluation metrics"""
        mode = 'train' if self.model.training else 'eval'
        device = self.accelerator.device

        # Calculate completion length metrics
        agg_completion_mask = gather(torch.cat([inp['completion_mask'].sum(1) for inp in inputs]))

        self._metrics[mode]['completions/mean_length'].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]['completions/min_length'].append(agg_completion_mask.float().min().item())
        self._metrics[mode]['completions/max_length'].append(agg_completion_mask.float().max().item())
        # Calculate clip ratio
        agg_truncated_mask = gather(torch.cat([inp['truncated_mask'] for inp in inputs]).to(device))

        term_completion_mask = agg_completion_mask[agg_truncated_mask]
        clipped_completions_ratio = len(term_completion_mask) / len(agg_completion_mask)

        self._metrics[mode]['completions/clipped_ratio'].append(clipped_completions_ratio)

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = rewards_per_func[:, i].mean().item()
            self._metrics[mode][f'rewards/{reward_func_name}/mean'].append(mean_rewards)
            std_rewards = rewards_per_func[:, i].std().item()
            self._metrics[mode][f'rewards/{reward_func_name}/std'].append(std_rewards)

        # Log overall reward stats
        grouped_rewards = rewards.view(-1, self.num_generations)
        self._metrics[mode]['reward'].append(grouped_rewards.mean().item())
        self._metrics[mode]['reward_std'].append(grouped_rewards.std(dim=1).mean().item())

        # Log prompt and completion texts
        self._textual_logs['prompt'].extend(self._apply_chat_template_to_messages_list(gather_object(messages)))
        self._textual_logs['completion'].extend(gather_object(completions))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs['rewards'][name].extend(rewards_per_func[:, i].tolist())

    def _apply_chat_template_to_messages_list(self, messages_list: InputsType):
        prompts_text = []
        for messages in messages_list:
            InferRequest.remove_response(messages)
            template_inputs = StdTemplateInputs.from_dict({'messages': messages})
            res_context_list, _, _ = self.template._swift_encode(template_inputs)
            prompts_text.append(''.join(res_context_list))
        return prompts_text

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute the per-token log probabilities for the model, return_outputs=True in mini-batch training
        if isinstance(inputs, list):
            assert len(inputs) == 1
            inputs = inputs[0]
        completion_mask = inputs['completion_mask']
        truncated_mask = inputs['truncated_mask']
        # apply the completion_mask to exclude loss and metrics for overlong completions
        if self.args.overlong_filter and any(truncated_mask):
            if all(truncated_mask):
                logger.info('All completions are overlong, loss and KL will be zero')
            truncated_mask = truncated_mask.unsqueeze(-1).expand_as(completion_mask).to(completion_mask.device)
            completion_mask = completion_mask * (~truncated_mask)

        per_token_logps = self._get_per_token_logps(model, inputs)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs['ref_per_token_logps']
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1)

        advantages = inputs['advantages']
        old_per_token_logps = inputs['old_per_token_logps'] if self.old_policy else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == 'grpo':
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == 'bnpo':
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == 'dr_grpo':
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        # Log the metrics
        mode = 'train' if self.model.training else 'eval'

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]['kl'].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]['clip_ratio/low_mean'].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]['clip_ratio/low_min'].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]['clip_ratio/high_mean'].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]['clip_ratio/high_max'].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]['clip_ratio/region_mean'].append(gathered_clip_ratio.nanmean().item())

        return loss

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, inputs):
        from trl.trainer.utils import selective_log_softmax
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        unwrapped_model = self.accelerator.unwrap_model(model)
        if is_peft_model(unwrapped_model):
            parameters = inspect.signature(unwrapped_model.base_model.model.forward).parameters
        else:
            parameters = inspect.signature(unwrapped_model.forward).parameters
        if not unwrapped_model.model_meta.is_multimodal and 'logits_to_keep' in parameters:
            # save memory
            return super()._get_per_token_logps(model, input_ids, inputs['attention_mask'], logits_to_keep)
        inputs = {
            k: v
            for k, v in inputs.items() if k not in [
                'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps',
                'truncated_mask'
            ]
        }
        with self._template_context(self.template):
            logits = model(**inputs).logits
        # exclude the last logit: it corresponds to the next token pred
        logits = logits[:, -(logits_to_keep + 1):-1, :]
        logits = logits / self.temperature
        input_ids = input_ids[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def evaluation_loop(self, dataloader, *args, **kwargs):
        # Wait for the training rollout to complete
        if self.args.async_generate:
            while not self.is_async_generate_train_rollout_done():
                time.sleep(0.1)
        if self._queue.empty() and self.args.async_generate:
            self._prefetch(dataloader)
        metric_key_prefix = kwargs['metric_key_prefix']
        output = super().evaluation_loop(dataloader, *args, **kwargs)
        metrics = {f'{metric_key_prefix}_{key}': sum(val) / len(val) for key, val in self._metrics['eval'].items()}
        output.metrics.update(metrics)
        self.eval_flag = True
        return output

    def training_step(self, model: nn.Module, inputs: InputsType, num_items_in_batch=None) -> torch.Tensor:
        if self.args.async_generate:
            # Wait for the eval rollout to complete
            while not self.is_async_generate_eval_rollout_done():
                time.sleep(0.1)
        return super().training_step(model, inputs, num_items_in_batch)

    def _engine_infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        *,
        use_tqdm: Optional[bool] = False,
    ):
        if self.vllm_mode == 'server':
            self._process_infer_requests_images(infer_requests)
            return self.vllm_client.infer(infer_requests, asdict(request_config), use_tqdm=use_tqdm)
        else:
            # return self.engine.infer(infer_requests, request_config, use_tqdm=use_tqdm)
            return self.engine.infer(infer_requests, request_config, use_tqdm=use_tqdm, process_index=self.accelerator.process_index)

    def _process_infer_requests_images(self, infer_requests: List[InferRequest]):
        # Process image format into a format that session.post can accept
        import base64
        if not any('images' in request for request in infer_requests):
            return
        for request in infer_requests:
            if 'images' not in request:
                continue
            for i, img in enumerate(request['images']):
                if 'bytes' in img and img['bytes']:
                    request['images'][i] = base64.b64encode(img['bytes']).decode('utf-8')
                elif 'path' in img and img['path']:
                    request['images'][i] = img['path']
        return

    @property
    def old_policy(self):
        return self.num_iterations > 1

    @property
    def _queue(self):
        if self.control.should_evaluate:
            return self.eval_queue
        else:
            return self.train_queue

    @torch.no_grad()
    def offload_model(self):
        if len(self.offload_modules) > 0:
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for name, module in unwrapped_model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                self.offload_modules[name] = module.weight.device
                module.to('cpu')
            elif not hasattr(module, 'device'):
                pass
            elif module.device.type != 'cpu':
                self.offload_modules[name] = module.device
                module.to('cpu')

    @torch.no_grad()
    def load_model(self):
        if len(self.offload_modules) == 0:
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for name, device in self.offload_modules.items():
            module = unwrapped_model.get_submodule(name)
            if isinstance(module, torch.nn.Embedding):
                module.weight.to(device)
            else:
                module.to(device)
        self.offload_modules.clear()

    @torch.no_grad()
    def offload_optimizer(self):
        if len(self.offload_states) > 0:
            return
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        self.offload_states[key] = value.device
                        state[key] = value.to('cpu', non_blocking=True)

    @torch.no_grad()
    def load_optimizer(self):
        if len(self.offload_states) == 0:
            return
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self.offload_states[key], non_blocking=True)
        self.offload_states.clear()

    @contextmanager
    def multi_turn_completion_length_context(self):
        """
        Context manager that temporarily adjusts the engine's max length handling
        for multi-turn generation scenarios.

        Ensures the total sequence length (prompt + completion) never exceeds:
            min(original_max_len, prompt_tokens + max_completion_length)
        """
        if not (self.multi_turn_func and
                self.use_fast_infer) or self.vllm_mode == 'server' or self.completion_length_limit_scope == 'per_round':
            yield
            return

        original_fn = self.engine.set_default_max_tokens
        original_max_len = self.engine.max_model_len

        def set_default_max_tokens(_self, request_config: RequestConfig, inputs: InputsType) -> None:
            # Calculate required context window
            original_max_len = _self.max_model_len or 8192
            if isinstance(inputs, dict):
                inputs = [inputs]
            prompt_tokens = max(_self._get_num_tokens(inp) for inp in inputs)

            if not hasattr(_self, 'set_grpo_max_model_len'):
                # set max model len in first round
                max_len = min(original_max_len, prompt_tokens + request_config.max_tokens)
                _self.max_model_len = max_len
                _self.set_grpo_max_model_len = True
            else:
                if _self.max_model_len <= prompt_tokens:
                    # modify max_model_len > prompt_tokens to avoid crash
                    num_tokens_avoid_crash = 10
                    _self.max_model_len = (prompt_tokens + num_tokens_avoid_crash)
                    request_config.max_tokens = num_tokens_avoid_crash

            original_fn(request_config, inputs)

        try:
            self.engine.set_default_max_tokens = MethodType(set_default_max_tokens, self.engine)
            yield
        finally:
            self.engine.set_default_max_tokens = original_fn
            self.engine.max_model_len = original_max_len
            del self.engine.set_grpo_max_model_len

    def get_resample_dataloader(self) -> DataLoader:
        resample_dataset = self.resample_dataset
        data_collator = self.data_collator
        if isinstance(resample_dataset, datasets.Dataset):
            resample_dataset = self._remove_unused_columns(resample_dataset, description='training')
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description='training')

        dataloader_params = {
            'batch_size': self._train_batch_size * self.args.gradient_accumulation_steps,
            'collate_fn': data_collator,
            'num_workers': self.args.dataloader_num_workers,
            'pin_memory': self.args.dataloader_pin_memory,
            'persistent_workers': self.args.dataloader_persistent_workers,
        }

        @contextmanager
        def seed_context(self):
            seed = self.args.seed
            self.args.seed = seed + 1
            yield
            self.args.seed = seed

        if not isinstance(resample_dataset, torch.utils.data.IterableDataset):
            with seed_context(self):  # Set a different seed for resampling than the train_dataset.
                dataloader_params['sampler'] = self._get_train_sampler()
            dataloader_params['drop_last'] = self.args.dataloader_drop_last
            dataloader_params['worker_init_fn'] = seed_worker
            dataloader_params['prefetch_factor'] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(resample_dataset, **dataloader_params))

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = 'train' if self.model.training else 'eval'
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == 'eval':
            metrics = {f'eval_{key}': val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            table = {
                'step': [str(self.state.global_step)] * len(self._textual_logs['prompt']),
                'prompt': self._textual_logs['prompt'],
                'completion': self._textual_logs['completion'],
                **self._textual_logs['rewards'],
            }
            self.jsonl_writer.append(table)
            if self.args.report_to and 'wandb' in self.args.report_to and wandb.run is not None:
                import pandas as pd
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                wandb.log({'completions': wandb.Table(dataframe=df)})

    def is_async_generate_eval_rollout_done(self):
        return not self.eval_flag or not self.eval_queue.empty()

    def is_async_generate_train_rollout_done(self):
        return not self.train_queue.empty()
