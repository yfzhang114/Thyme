from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import sys
import warnings
import math
import logging
import copy
import torch
import shutil
import json
from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag, listinstr
from .sandbox import execute_code_in_sandbox
from .utils import (generate_prompt_final_qa, generate_prompt_simple_qa, SPECIAL_STRING_LIST, 
                    REASONING_SYS_PROMPT, SIMPLE_SYS_PROMPT
                    )
import re
import random

from transformers.cache_utils import (
    DynamicCache
)

CACHE_DIR_PATH="./vis_cases_max_2048"

def write_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }


def encode_image(image_path, max_side=None):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    from PIL import Image
    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type


def _encode_image(image, image_format):
    from io import BytesIO
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        import base64
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data


def _rgba_to_rgb(image):
    from PIL import Image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)


def process_video(video_path, num_frames, min_pixels, max_pixels):
    import cv2
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # the sampling rate using max number of frames
    sampling_gap_maxframe = (
        1 if not num_frames else math.ceil(frame_count / num_frames)
    )
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        import tempfile
        success, frame = cap.read()
        if not success:
            break
        # Sample frames based on the dynamic sampling rate
        if frame_number % sampling_gap == 0:
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(
        f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
    )
    cap.release()
    return images


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map


def setup_visible_devices_per_rank():
    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    assert world_size == 1, "Only support world_size == 1 for vLLM inference"
    num_gpus = total_gpus // world_size
    start_idx = rank * num_gpus
    assigned_devices = list(range(start_idx, start_idx + num_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in assigned_devices)
    logging.info(f"[Rank {rank}] Visible GPUs: {assigned_devices}")
    return num_gpus


class QwenTool(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        max_iterations = 5,     # pandayin: rounds of intermediate steps before reaching final answer.
        max_retry =5,           # pandayin: max retry before reaching a valid answer.
        use_custom_prompt: bool = True,
        system_prompt: str | None = "You are a helpful assistant.",
        post_process: bool = True,  # if True, will try to only extract stuff wrapped in <answer> & </answer>.
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_retry = max_retry
        self.verbose = verbose
        self.post_process = post_process
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None

        # if listinstr(['omni'], model_path.lower()):
        #     try:
        #         from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        #     except Exception as err:
        #         logging.critical("pip install git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356")  # noqa: E501
        #         raise err
        #     MODEL_CLS = Qwen2_5OmniForConditionalGeneration
        #     self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        # elif listinstr(['2.5', '2_5', 'qwen25', 'tool'], model_path.lower()):
        #     from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        #     MODEL_CLS = Qwen2_5_VLForConditionalGeneration
        #     self.processor = AutoProcessor.from_pretrained(model_path)
        # else:
        #     from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        #     MODEL_CLS = Qwen2VLForConditionalGeneration
        #     self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        # pandayin: hard-coding since we only use Qwen2.5-VL as base models for now.
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        MODEL_CLS = Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path)
            
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            stop_strings=SPECIAL_STRING_LIST,
            tokenizer=self.processor.tokenizer
        )
        
        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0
        self.use_vllm = kwargs.get('use_vllm', False)
        self.limit_mm_per_prompt = 24
        if self.use_vllm:
            from vllm import LLM
            gpu_count = setup_visible_devices_per_rank()
            if gpu_count >= 8:
                tp_size = 7
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            import os
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    'VLLM_WORKER_MULTIPROC_METHOD is not set to spawn.'
                    'Use \'export VLLM_WORKER_MULTIPROC_METHOD=spawn\' to avoid potential multi-process issues'
                )

            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=5,
                max_model_len=32768,
                limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
                enable_prefix_caching=True  # use prefix caching to save inference time.
            )

        else:
            # If only one process and GPU memory is less than 40GB
            if '72b' in self.model_path.lower():
                self.model = MODEL_CLS.from_pretrained(
                    model_path, torch_dtype='auto', device_map=split_model(), attn_implementation='flash_attention_2'
                )
                self.model.eval()
            elif auto_split_flag():
                assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
                # Will Use All GPUs to run one model
                self.model = MODEL_CLS.from_pretrained(
                    model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            else:
                self.model = MODEL_CLS.from_pretrained(
                    model_path, torch_dtype='auto', device_map='cuda', attn_implementation='flash_attention_2'
                )
                self.model.eval()

        torch.cuda.empty_cache()

    def _extract_image_path(self, contents: list[dict[str, str]]):
        user_image_path = ""
        content_history = copy.deepcopy(contents)
        for rou in content_history:
            if rou['type'] != 'image': continue
            user_image_path = rou['value']
            break
        return user_image_path
    
    def _extract_question(self, contents: list[dict[str, str]]) -> str:
        qs = ""
        content_history = copy.deepcopy(contents)
        for rou in content_history:
            if rou['type'] != 'text': continue
            qs = rou['value']
            break
        return qs
    
    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        user_image_path = self._extract_image_path(inputs)
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': ensure_video_url(s['value']),
                    'min_pixels': self.min_pixels,
                    'max_pixels': self.max_pixels
                }
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            # pandayin: wrap user query with customized prompt.
            # TODO: consider support for multi-images or videos...
            elif s['type'] == 'text':
                #item = {'type': 'text', 'text': s['value']}
                item = {'type': 'text', 'text': generate_prompt_final_qa(s['value'], user_image_path)}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _prepare_content_simple(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': ensure_video_url(s['value']),
                    'min_pixels': self.min_pixels,
                    'max_pixels': self.max_pixels
                }
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            # pandayin: wrap user query with customized prompt.
            # TODO: consider support for multi-images or videos...
            elif s['type'] == 'text':
                #item = {'type': 'text', 'text': s['value']}
                item = {'type': 'text', 'text': generate_prompt_simple_qa(s['value'])}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content
    
    def _prepare_content_vllm(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        user_image_path = self._extract_image_path(inputs)
        content = []
        video_inputs = [s for s in inputs if s['type'] == 'video']
        video_count = len(video_inputs)
        cur_image_count = 0
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if cur_image_count < self.limit_mm_per_prompt:
                    content.append(item)
                    cur_image_count += 1
                else:
                    logging.warning(
                        f"Number of images exceeds the limit of {self.limit_mm_per_prompt}. "
                        f"Only the first {self.limit_mm_per_prompt} images will be used."
                    )
            elif s['type'] == 'video':
                if video_count > 1:
                    logging.warning(
                        "Multiple videos detected. Using video frames for each video"
                    )
                    if dataset == 'OCRBench':
                        min_pixels = 10 * 10 * 28 * 28
                        warnings.warn(f"OCRBench dataset uses custom min_pixels={min_pixels}")
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    else:
                        if self.min_pixels is not None:
                            min_pixels = self.min_pixels
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()

                    frames_per_video = max(1, self.limit_mm_per_prompt // video_count)
                    content.append({"type": "text", "text": "<video frames start>"})
                    content.extend(process_video(s['value'], frames_per_video, min_pixels, max_pixels))
                    content.append({"type": "text", "text": "<video frames end>"})

                else:
                    item = {
                        'type': 'video',
                        'video': ensure_video_url(s['value']),
                        'min_pixels': self.min_pixels,
                        'max_pixels': self.max_pixels
                    }
                    if self.fps is not None:
                        item['fps'] = self.fps
                    elif self.nframe is not None:
                        import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                        content.append(item)
                    else:
                        item['nframes'] = self.nframe
                        content.append(item)
            elif s['type'] == 'text':
                #item = {'type': 'text', 'text': s['value']}
                item = {'type': 'text', 'text': generate_prompt_final_qa(s['value'], user_image_path)}
                content.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        return content

    def _prepare_content_vllm_simple(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        video_inputs = [s for s in inputs if s['type'] == 'video']
        video_count = len(video_inputs)
        cur_image_count = 0
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if cur_image_count < self.limit_mm_per_prompt:
                    content.append(item)
                    cur_image_count += 1
                else:
                    logging.warning(
                        f"Number of images exceeds the limit of {self.limit_mm_per_prompt}. "
                        f"Only the first {self.limit_mm_per_prompt} images will be used."
                    )
            elif s['type'] == 'video':
                if video_count > 1:
                    logging.warning(
                        "Multiple videos detected. Using video frames for each video"
                    )
                    if dataset == 'OCRBench':
                        min_pixels = 10 * 10 * 28 * 28
                        warnings.warn(f"OCRBench dataset uses custom min_pixels={min_pixels}")
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    else:
                        if self.min_pixels is not None:
                            min_pixels = self.min_pixels
                        if self.max_pixels is not None:
                            max_pixels = self.max_pixels
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()

                    frames_per_video = max(1, self.limit_mm_per_prompt // video_count)
                    content.append({"type": "text", "text": "<video frames start>"})
                    content.extend(process_video(s['value'], frames_per_video, min_pixels, max_pixels))
                    content.append({"type": "text", "text": "<video frames end>"})

                else:
                    item = {
                        'type': 'video',
                        'video': ensure_video_url(s['value']),
                        'min_pixels': self.min_pixels,
                        'max_pixels': self.max_pixels
                    }
                    if self.fps is not None:
                        item['fps'] = self.fps
                    elif self.nframe is not None:
                        import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                        content.append(item)
                    else:
                        item['nframes'] = self.nframe
                        content.append(item)
            elif s['type'] == 'text':
                #item = {'type': 'text', 'text': s['value']}
                item = {'type': 'text', 'text': generate_prompt_simple_qa(s['value'])}
                content.append(item)
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        return content
    
    def _extract_box_answer(self, response):
        resp = response.split('\\boxed{')[-1]
        lt = len(resp)
        counter, end = 1, None
        for i in range(lt):
            if resp[i] == '{':
                counter += 1
            elif resp[i] == '}':
                counter -= 1
            if counter == 0:
                end = i
                break
            elif i == lt - 1:
                end = lt
                break
        if end is not None:
            response = resp[:end]
        return response
    
    def _remove_unpickable_values(self, dictionary):
        import pickle

        def is_pickable(obj):
            try:
                pickle.dumps(obj)
                return True
            except (pickle.PicklingError, TypeError, AttributeError):
                return False
    
        keys_to_remove = []
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self._remove_unpickable_values(value)
            elif not is_pickable(value):
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del dictionary[key]
        return dictionary
        

    def generate_inner_transformers(self, message, dataset=None):
        if listinstr(['omni'], self.model_path.lower()):
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
                raise err
        # pandayin: For ease of visualization, I insert the label/GT as the first line of message.. So strip it.
        ground_truth = message[0]['value']
        message = message[1:]
        
        # pandayin: get image path from the input sample.
        user_image_path = self._extract_image_path(message)
        user_query = self._extract_question(message)
        # TODO: add arg to optionally select this behaviour.
        # save whole process for visualization.
        case_id, case_ext = os.path.splitext(os.path.basename(user_image_path))
        case_save_dir = os.path.join(CACHE_DIR_PATH, dataset, case_id)
        os.makedirs(case_save_dir, exist_ok=True)
        # copy the original img.
        shutil.copy(user_image_path, case_save_dir)
        
        messages = []
        # if self.system_prompt is not None:
        #     messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'system', 'content': REASONING_SYS_PROMPT})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        
        
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        # text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        # if listinstr(['omni'], self.model_path.lower()):
        #     _, images, videos = process_mm_info([messages], use_audio_in_video=False)
        # else:
        #     images, videos = process_vision_info([messages])
        # inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        # inputs = inputs.to('cuda')

        # if listinstr(['omni'], self.model_path.lower()):
        #     self.generate_kwargs['use_audio_in_video'] = False
        #     self.generate_kwargs['return_audio'] = False
            
        
        # pandayin: add fail generation log for ease of analysis.
        fail_cases = []
        # pandayin: 
        #   -------   outer loop. retry multiple times if fail to reach a valid answer.   -------
        retry_generations = self.max_retry
        has_valid_answer = False
        while (retry_generations > 0) and (not has_valid_answer):
            # pandayin: main logic/ work flow for generation. 
            # The gist is to pause at special tokens (</code> & </answer>) and maybe perform code execution.
            # TODO: maybe abstract this into an independent function 
            conversation_history = copy.deepcopy(messages)
            
            # For each generation, we initialize a KV-Cache to speed up inference.
            kv_cache = DynamicCache()
            # Maintain a dictionary to save context (local & global vars.) for code execution.
            previous_execution_context = {}
            if self.verbose:
                print(f'\033[32m\n--- Generation {self.max_retry - retry_generations + 1} ---\033[0m')
                    
            # pandayin: 
            #   -------   inner loop. generate multiple steps until reaching a valid answer.   -------
            retry_iterations = self.max_iterations
            # We assume each answer round is limited to a few code (usually 1) execution.
            while retry_iterations > 0:
                retry_iterations -= 1
                generated_content = []
                if self.verbose:
                    print(f'\033[32m\n--- Iteration {self.max_iterations - retry_iterations} ---\033[0m')
                
                
                text = self.processor.apply_chat_template([conversation_history], 
                                                          tokenize=False, 
                                                          add_generation_prompt=(retry_iterations==self.max_iterations-1)
                                                          )
                if retry_iterations != self.max_iterations-1:
                    if text[0].endswith("<|im_end|>\n"):
                        text[0] = text[0][:-len("<|im_end|>\n")]

                # import pdb; pdb.set_trace()
                if listinstr(['omni'], self.model_path.lower()):
                    _, images, videos = process_mm_info([conversation_history], use_audio_in_video=False)
                else:
                    images, videos = process_vision_info([conversation_history])
                inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
                inputs = inputs.to('cuda')

                if listinstr(['omni'], self.model_path.lower()):
                    self.generate_kwargs['use_audio_in_video'] = False
                    self.generate_kwargs['return_audio'] = False
                
                # just in case this iteration is invalid, we need to roll back, thus making a backup.
                last_kv_cache = copy.deepcopy(kv_cache)
                # bkup context. roll back when we fail to execute the generated code.
                last_execution_context = copy.deepcopy(self._remove_unpickable_values(previous_execution_context))  #copy.deepcopy(previous_execution_context)
                generated_ids = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                    past_key_values=kv_cache
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                out = self.processor.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                generated_text_segment = out[0]
                
                # Case 1: directly give answer
                if "</answer>" in generated_text_segment:
                    generated_content.append(
                        {"type": "text", "text": generated_text_segment},
                    )

                # import pdb; pdb.set_trace()
                
                # Case 2: reach code generation.
                # parse current result. Two cases: reach </code> or reach </answer>
                code_regex = re.compile(r'<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>', re.IGNORECASE)
                #re.compile(r'<code>\s*(?:python\s*)?([\s\S]*?)\s*</code>')

                # generated_text_segment = '<code>The image shows two men engaged in a conversation. One man is wearing a white shirt and the other is wearing a striped shirt. The man in the white shirt appears to be speaking, while the man in the striped shirt is listening attentively. The background is blurred, but it seems to be an indoor setting with some furniture visible. The image also includes a watermark that reads "SPENCER TRACY".</code>'


                code_match = code_regex.search(generated_text_segment)
                
                # execute code and return result.
                if code_match:
                    code_to_execute = code_match.group(1).strip()
                    if self.verbose:
                        print(f"\033[31m--- Found Code Block ---\n{generated_text_segment}\n-------------------------\033[0m")

                    # pandayin: create additional dirs to save intermediate images..
                    # since sometimes image path from stdout does not match the exact files....  
                    intermediate_images_save_dir = os.path.join(case_save_dir, 
                        f"Gen-{self.max_retry - retry_generations + 1}-Iter-{self.max_iterations - retry_iterations}")
                    os.makedirs(intermediate_images_save_dir, exist_ok=True)
                    
                    has_valid_images = False
                    processed_img_paths, captured_stdout, error_msg, current_execution_context = execute_code_in_sandbox(
                        code_to_execute, user_image_path, temp_output_dir=intermediate_images_save_dir,
                        previous_execution_context=previous_execution_context
                    )
                    previous_execution_context = current_execution_context
                    # import pdb; pdb.set_trace()
                    if not processed_img_paths:
                        kv_cache = last_kv_cache    # deemed as unsuccessful iteration. roll back status.
                        previous_execution_context = last_execution_context
                        print(f'{error_msg}')
                        fail_cases.append({'wrong code': code_to_execute})
                        continue      
                       
                    has_valid_images = False
                    generated_content += [
                                        {"type": "text", "text": generated_text_segment},
                                        {"type": "text", "text": "<sandbox_output>"}
                                    ]
                    first_path = processed_img_paths[0]
                    if os.path.exists(first_path):
                        # Iterate through each path in the list
                        for img_path in processed_img_paths:
                            if os.path.exists(img_path):
                                if not has_valid_images: # Add text segments only once per sandbox output block
                                    has_valid_images = True
                                generated_content.append({"type": "image", "image": img_path})                            
                    else:
                        generated_content.append({"type": "text", "text": first_path})

                    if has_valid_images or not os.path.exists(first_path):
                        generated_content.append({"type": "text", "text": "</sandbox_output>"})
                    else:
                        # pandayin: a failed code execution/generation doesn't count as a intermedia step.
                        print(processed_img_paths[0])
                        print('skip this generation due to error and adapt the temperature')
                        self.generate_kwargs['temperature'] = 1.0
                        continue
                else:
                    # wo code. wo </answer>, 那么基本上一直在生成重复内容，直接break这次
                    if "</answer>" not in generated_text_segment:
                        print('wo code. wo </answer>')
                        print(generated_text_segment)
                        fail_cases.append({'no code nor answer': generated_text_segment})
                        self.generate_kwargs['temperature'] = 1.0
                        # retry_generations -= 1
                        break

                # Update conversation_history with the latest generated segment
                # If the last message was 'user', start a new 'assistant' message
                if conversation_history[-1]["role"] == "user":
                    conversation_history.append({"role": "assistant", "content": generated_content})
                # If the last message was 'assistant', append to its last text content item
                elif conversation_history[-1]["role"] == "assistant":
                    conversation_history[-1]["content"] += generated_content
                
                # --- Check for final answer tag if no code was processed in this segment ---
                if "</answer>" in generated_text_segment:
                    has_valid_answer = True
                    print("\033[32m--- Final answer tag found. ---\033[0m")
                    break
                
                # If the model produced an EOS token and no code/answer, it might be finished
                if generated_ids[0][-1] == self.processor.tokenizer.eos_token_id:
                    if self.verbose:
                        print("\033[32m--- Model generated EOS and no further actions (code/answer). Assuming completion. ---\033[0m")
                    break
            
                #retry_iterations -= 1
            
            # End of a generation. Maybe successfully find a valid answer, or start a new generation.
            if self.verbose:
                if has_valid_answer:
                    print(f"\033[32m\n--- End of processing (max iterations: {self.max_iterations}, actual: {self.max_iterations - retry_iterations + 1}) ---\033[0m")
                    break
                else:
                    print(f"\033[32m\n --- Fail to find a valid answer. (max retrys: {self.max_retry}, actual: {self.max_retry - retry_generations + 1})---\033[0m")
            
            retry_generations -= 1 
            # pandayin: TODO: maybe adjust/reset generation_kwargs here? So more explorations could be done to find a valid answer. 
            print('Fail to find a valid answer and adapt the temperature')
            self.generate_kwargs['temperature'] = 1.0

        
        # reset generation hyper-param.
        self.generate_kwargs['temperature'] = 0.01
        
        # pandayin: If we still fail after max_try generations, try a simple prompt.
        if not has_valid_answer:
            print(f"\033[32m\n --- Fail to find a valid answer after {self.max_retry} retrys. Falling back to simple prompt.---\033[0m")
            del self.generate_kwargs['stop_strings']
            
            messages = []
            if self.system_prompt is not None:
                messages.append({'role': 'system', 'content': SIMPLE_SYS_PROMPT})
            messages.append({'role': 'user', 'content': self._prepare_content_simple(message, dataset=dataset)})
            conversation_history = copy.deepcopy(messages)
            text = self.processor.apply_chat_template([conversation_history], 
                                                          tokenize=False, 
                                                          add_generation_prompt=True
                                                          )
            if listinstr(['omni'], self.model_path.lower()):
                    _, images, videos = process_mm_info([conversation_history], use_audio_in_video=False)
            else:
                images, videos = process_vision_info([conversation_history])
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
            inputs = inputs.to('cuda')
            generated_ids = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            out = self.processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            generated_text_segment = out[0]
            
            self.generate_kwargs['stop_strings'] = SPECIAL_STRING_LIST
            
            # to align with the following processing procedure. wrap a <answer> bracket.
            answer_match = re.search(r"<answer>(.*?)</answer>", generated_text_segment, re.DOTALL)
            if not answer_match:
                generated_text_segment = "<answer>" + generated_text_segment + "</answer>"
            conversation_history.append({"role": "assistant", "content": [{"type": "text", "text": generated_text_segment}]})
            
            
        
        final_assistant_response = ""
        for msg in reversed(conversation_history):
            if msg['role'] != 'assistant': continue
            current_content_str = ""
            for item in msg['content']:
                if item['type'] == 'text':
                    current_content_str += item['text']
            final_assistant_response = current_content_str # Get the last full response from assistant
            break
        
          
        if self.post_process:
            print(f"\033[31m--- Final response ---\n{final_assistant_response}\n-------------------------\033[0m")
            # Extract content within <answer> tags from the final assistant response
            answer_match = re.search(r"<answer>(.*?)</answer>", final_assistant_response, re.DOTALL)
            if answer_match:
                final_answer = answer_match.group(1).strip()
            else:
                fail_cases.append({'unclosed answer': final_assistant_response})
                final_answer = "No answer tag found in the final output."

            # Sometimes the answer is still wrapped in \boxed{}, keeping the behaviour of Qwen2.5-VL.
            # We extract the answer within this.
            match = re.search(r'\\boxed\{(.*?)\}', final_answer)
            if match:
                final_answer = self._extract_box_answer(final_answer)
                #final_answer = match.group(1).strip()
            

            if self.verbose:
                print(f'\033[32m{final_answer}\033[0m')
                
            
            # cache final full answer in a json.
            final_results = {
                "qs": user_query,
                "orig_img_path": os.path.basename(user_image_path),
                "final_answer": final_answer,
                "ground_truth": ground_truth,
                "trajectory": conversation_history[1:],
                "fail_cases": fail_cases
            }
            write_json(os.path.join(case_save_dir, "vis.json"), final_results)
            return final_answer
        else:
            return final_assistant_response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams

        # if listinstr(['omni'], self.model_path.lower()):
        #     try:
        #         from qwen_omni_utils import process_mm_info
        #     except Exception as err:
        #         logging.critical("qwen_omni_utils not found, please install it via 'pip install qwen-omni-utils[decord]'")  # noqa: E501
        #         raise err
        # else:
        #     try:
        #         from qwen_vl_utils import process_vision_info
        #     except Exception as err:
        #         logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
        #         raise err
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")  # noqa: E501
            raise err

        # pandayin: For ease of visualization, I insert the label/GT as the first line of message.. So strip it.
        ground_truth = message[0]['value']
        message = message[1:]
        
        # pandayin: get image path from the input sample.
        user_image_path = self._extract_image_path(message)
        user_query = self._extract_question(message)
        # TODO: add arg to optionally select this behaviour.
        # save whole process for visualization.
        case_id, case_ext = os.path.splitext(os.path.basename(user_image_path))
        case_save_dir = os.path.join(CACHE_DIR_PATH, dataset, case_id)
        os.makedirs(case_save_dir, exist_ok=True)
        # copy the original img.
        shutil.copy(user_image_path, case_save_dir)
        
        messages = []
        # if self.system_prompt is not None:
        #     messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'system', 'content': REASONING_SYS_PROMPT})
        messages.append({'role': 'user', 'content': self._prepare_content_vllm(message, dataset=dataset)})
        
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')
        
        # pandayin: add fail generation log for ease of analysis.
        fail_cases = []
        # pandayin: 
        #   -------   outer loop. retry multiple times if fail to reach a valid answer.   -------
        retry_generations = self.max_retry
        has_valid_answer = False
        while (retry_generations > 0) and (not has_valid_answer):
            retry_generations -= 1
            conversation_history = copy.deepcopy(messages)
            if self.verbose:
                print(f'\033[32m\n--- Generation {self.max_retry - retry_generations + 1} ---\033[0m')
            
            retry_iterations = self.max_iterations
            while retry_iterations > 0:
                retry_iterations -= 1
                generated_content = []
                if self.verbose:
                    print(f'\033[32m\n--- Iteration {self.max_iterations - retry_iterations} ---\033[0m')
                
                text = self.processor.apply_chat_template([conversation_history], tokenize=False, add_generation_prompt=True)
                if retry_iterations != self.max_iterations-1:
                    if text[0].endswith("<|im_end|>\n"):
                        text[0] = text[0][:-len("<|im_end|>\n")]
                    
                images, videos = process_vision_info(messages)
                
                sampling_params = SamplingParams(
                    temperature=self.temperature, max_tokens=2048, stop=SPECIAL_STRING_LIST, include_stop_str_in_output=True
                )
                if images:
                    outputs = self.llm.generate(
                        {
                            "prompt": text,
                            "multi_modal_data": {"image": images},
                        },
                        sampling_params=sampling_params,
                    )
                elif videos:
                    outputs = self.llm.generate(
                        {
                            "prompt": text,
                            "multi_modal_data": {"video": videos},
                        },
                        sampling_params=sampling_params,
                    )
                else:
                    outputs = self.llm.generate(
                        {
                            "prompt": text,
                        },
                        sampling_params=sampling_params,
                    )

                for o in outputs:
                    generated_text_segment = o.outputs[0].text
        
                # Case 1: directly give answer
                if "</answer>" in generated_text_segment:
                    generated_content.append(
                        {"type": "text", "text": generated_text_segment},
                    )

                # import pdb; pdb.set_trace()
                
                # Case 2: reach code generation.
                # parse current result. Two cases: reach </code> or reach </answer>
                code_regex = re.compile(r'<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>', re.IGNORECASE)
                #re.compile(r'<code>\s*(?:python\s*)?([\s\S]*?)\s*</code>')
                code_match = code_regex.search(generated_text_segment)
                
                # execute code and return result.
                if code_match:
                    code_to_execute = code_match.group(1).strip()
                    if self.verbose:
                        print(f"\033[31m--- Found Code Block ---\n{code_to_execute}\n-------------------------\033[0m")

                    # pandayin: create additional dirs to save intermediate images..
                    # since sometimes image path from stdout does not match the exact files....  
                    intermediate_images_save_dir = os.path.join(case_save_dir, 
                        f"Gen-{self.max_retry - retry_generations + 1}-Iter-{self.max_iterations - retry_iterations}")
                    os.makedirs(intermediate_images_save_dir, exist_ok=True)
                    
                    has_valid_images = False
                    processed_img_paths, captured_stdout, error_msg = execute_code_in_sandbox(
                        code_to_execute, user_image_path, temp_output_dir=intermediate_images_save_dir
                    )

                    if not processed_img_paths:
                        # import pdb; pdb.set_trace()
                        print(f'{error_msg}')
                        fail_cases.append({'wrong code': code_to_execute})
                        continue         
                    has_valid_images = False
                    generated_content += [
                                        {"type": "text", "text": generated_text_segment},
                                        {"type": "text", "text": "<sandbox_output>"}
                                    ]
                    first_path = processed_img_paths[0]
                    if os.path.exists(first_path):
                        # Iterate through each path in the list
                        for img_path in processed_img_paths:
                            if os.path.exists(img_path):
                                if not has_valid_images: # Add text segments only once per sandbox output block
                                    has_valid_images = True
                                generated_content.append({"type": "image", "image": img_path})                            
                    else:
                        generated_content.append({"type": "text", "text": first_path})

                    if has_valid_images or not os.path.exists(first_path):
                        generated_content.append({"type": "text", "text": "</sandbox_output>"})
                    else:
                        # pandayin: a failed code execution/generation doesn't count as a intermedia step.
                        print(processed_img_paths[0])
                        print('skip this generation due to error and adapt the temperature')
                        self.generate_kwargs['temperature'] = 1.0
                        continue
                else:
                    # wo code. wo </answer>, 那么基本上一直在生成重复内容，直接break这次
                    if "</answer>" not in generated_text_segment:
                        print('wo code. wo </answer>')
                        print(generated_text_segment)
                        fail_cases.append({'no code nor answer': generated_text_segment})
                        self.generate_kwargs['temperature'] = 1.0
                        retry_generations -= 1
                        break

                # Update conversation_history with the latest generated segment
                # If the last message was 'user', start a new 'assistant' message
                if conversation_history[-1]["role"] == "user":
                    conversation_history.append({"role": "assistant", "content": generated_content})
                # If the last message was 'assistant', append to its last text content item
                elif conversation_history[-1]["role"] == "assistant":
                    conversation_history[-1]["content"] += generated_content
                
                # --- Check for final answer tag if no code was processed in this segment ---
                if "</answer>" in generated_text_segment:
                    has_valid_answer = True
                    print("\033[32m--- Final answer tag found. ---\033[0m")
                    break
                
                # If the model produced an EOS token and no code/answer, it might be finished
                if generated_text_segment.endswith("<|im_end|>"):
                    generated_text_segment = generated_text_segment.rstrip("<|im_end|>")
                    if self.verbose:
                        print("\033[32m--- Model generated EOS and no further actions (code/answer). Assuming completion. ---\033[0m")
                    break
            
                #retry_iterations -= 1
            
            # End of a generation. Maybe successfully find a valid answer, or start a new generation.
            if self.verbose:
                if has_valid_answer:
                    print(f"\033[32m\n--- End of processing (max iterations: {self.max_iterations}, actual: {self.max_iterations - retry_iterations + 1}) ---\033[0m")
                    break
                else:
                    print(f"\033[32m\n --- Fail to find a valid answer. (max retrys: {self.max_retry}, actual: {self.max_retry - retry_generations + 1})---\033[0m")
            print('Fail to find a valid answer and adapt the temperature')
            self.temperature = 1.0
                
        
        self.temperature = 0.01
        # pandayin: If we still fail after max_try generations, try a simple prompt.
        if not has_valid_answer:
            print(f"\033[32m\n --- Fail to find a valid answer after {self.max_retry} retrys. Falling back to simple prompt.---\033[0m")
        
            messages = []
            messages.append({'role': 'system', 'content': SIMPLE_SYS_PROMPT})
            messages.append({'role': 'user', 'content': self._prepare_content_vllm_simple(message, dataset=dataset)})
            conversation_history = copy.deepcopy(messages)
            text = self.processor.apply_chat_template([conversation_history], tokenize=False, add_generation_prompt=True)
            
            images, videos = process_vision_info(messages)
                    
            sampling_params = SamplingParams(
                temperature=self.temperature, max_tokens=2048, stop_token_ids=None
            )
            
            if images:
                outputs = self.llm.generate(
                    {
                        "prompt": text,
                        "multi_modal_data": {"image": images},
                    },
                    sampling_params=sampling_params,
                )
            elif videos:
                outputs = self.llm.generate(
                    {
                        "prompt": text,
                        "multi_modal_data": {"video": videos},
                    },
                    sampling_params=sampling_params,
                )
            else:
                outputs = self.llm.generate(
                    {
                        "prompt": text,
                    },
                    sampling_params=sampling_params,
                )
            
            for o in outputs:
                generated_text_segment = o.outputs[0].text
                        
            # to align with the following processing procedure. wrap a <answer> bracket.
            answer_match = re.search(r"<answer>(.*?)</answer>", generated_text_segment, re.DOTALL)
            if not answer_match:
                generated_text_segment = "<answer>" + generated_text_segment + "</answer>"
            conversation_history.append({"role": "assistant", "content": [{"type": "text", "text": generated_text_segment}]})
        
        
        final_assistant_response = ""
        for msg in reversed(conversation_history):
            if msg['role'] != 'assistant': continue
            current_content_str = ""
            for item in msg['content']:
                if item['type'] == 'text':
                    current_content_str += item['text']
            final_assistant_response = current_content_str # Get the last full response from assistant
            break
        
        
        if self.post_process:
            print(f"\033[31m--- Final response ---\n{final_assistant_response}\n-------------------------\033[0m")
            # Extract content within <answer> tags from the final assistant response
            answer_match = re.search(r"<answer>(.*?)</answer>", final_assistant_response, re.DOTALL)
            if answer_match:
                final_answer = answer_match.group(1).strip()
            else:
                fail_cases.append({'unclosed answer': final_assistant_response})
                final_answer = "No answer tag found in the final output."

            # Sometimes the answer is still wrapped in \boxed{}, keeping the behaviour of Qwen2.5-VL.
            # We extract the answer within this.
            match = re.search(r'\\boxed\{(.*?)\}', final_answer)
            if match:
                final_answer = match.group(1).strip()
                
            if self.verbose:
                print(f'\033[32m{final_answer}\033[0m')
                
            
            # cache final full answer in a json.
            final_results = {
                "qs": user_query,
                "orig_img_path": os.path.basename(user_image_path),
                "final_answer": final_answer,
                "ground_truth": ground_truth,
                "trajectory": conversation_history[1:],
                "fail_cases": fail_cases
            }
            write_json(os.path.join(case_save_dir, "vis.json"), final_results)
            return final_answer
        else:
            return final_assistant_response

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)
