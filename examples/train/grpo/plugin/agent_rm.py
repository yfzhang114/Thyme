import re
import json
import base64
from typing import List
import os
import httpx
# from openai import OpenAI

import asyncio
from swift.plugin import ORM, orms
from swift.utils import get_logger
import numpy as np
from latex2sympy2_extended import NormalizationConfig
import timeout_decorator    # For more efficient timeout control
from swift.plugin.strict_math_verify import evaluate_math, verify_choice_answer
from math_verify import LatexExtractionConfig, parse, verify
import random
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
reward_ip_list = [
    "",
    "",
    "",
    "",
]

INVALID_REWARD_VALUE = 0 #-100

logger = get_logger()

code_fail_indicator = "Code execution failed"


# @timeout_decorator.timeout(10, use_signals=True)
async def llm_openai_api(
    messages,
    ip="0.0.0.0",
    host="8080",
    temperature=0.1,
    max_tokens=256,
    top_p=None,
    # openai_api_key="EMPTY",
    n=1,
):
    openai_api_base = f"http://{ip}:{host}/v1"
    # 使用异步 HTTP 客户端
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        model = "/mllm_hdd/yfzhang/models/Qwen2.5-VL-72B-Instruct-AWQ"
        resp = await client.post(
            f"{openai_api_base}/chat/completions",
            headers = {"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "n": n,
            },
        )
        resp.raise_for_status()
        response_data = resp.json()
        return [choice["message"]["content"] for choice in response_data["choices"]]

vqa_orm_template = '''
You are an expert about question answering. I will provide a question, a generated answer to the question and the reference answer. Please evaluate the Correctness of the generated answer: If the generated answer is consistent with the reference answer, rate the Correctness as 1, else rate 0.
Here is the question, the generated answer for you to evaluate and the reference answer:\n

#### Question: 
input_question\n

#### Generated Answer: 
gen_answer\n

#### Reference Answer: 
ref_answer\n

### Output Format (strictly follow)

Please provide an integer score to indicate the Correctness. Output the score in a JSON dictionary with nothing else for easy processing, in this form: {"Correctness": score}.

Your Evaluation Result:
'''

vqa_prm_template = '''
You are an expert about visual question answering. Here is an image. I will provide a question about the image, a generated solution process to the question and the reference answer. Please evaluate the quality of the generated solution process from below dimensions:

1. **Process Correctness**: Please check whether the solution porcess is consistent with the content of the image, i.e., there are no hallucinations or wrong descriptions, and it is closely related to the question. Verify whether the solution process is logical and free of errors. Please rate the generated solution's thinking process from 1 to 10, with higher scores indicating higher quality of the thinking process.
2. **Text Quality**: Please check the text quality of the generated solution. A solution with high text quality should be well-organized and concise. The following situations are considered poor text quality: 1. The solution contains repetitive sentences. 2. The solution does not provide any answer for the question. Please rate the solution's text quality from 1 to 10, with higher scores indicating higher text quality of the solution.

Here is the question, the generated solution for you to evaluate and the reference answer:\n

####  Question:
\ninput_question


#### Reference Answer:
\nref_answer


#### Generated Solution:
\ngen_solution

### Output Format (strictly follow)
Please provide an integer score for each dimension. Output the scores in a JSON dictionary with nothing else for easy processing, in this form: {"Process Correctness": s1, "Text Quality": s2}.

Your Evaluation Result:
'''



code_prm_template = '''
You are an expert in visual question answering. An image is provided, along with a question about it, several cropped subimages from the original image, and a reference answer for the question. Your task is to evaluate the **Validity** of these cropped subimages according to the following rules:

1. **Sufficiency**: If any single subimage provides enough information to fully answer the question, then the Validity of *all* provided subimages is rated as **1**.
2. **Necessity**: If any single subimage contains information that is absolutely necessary to answer the question (meaning the question cannot be answered without it), then the Validity of *all* provided subimages is rated as **1**.
3. If all the provided cropped subimages are completely useless or unrelated to the question and its reference answer, then the Validity of these subimages is rated as **0**.

Here is the question and the reference answer for your evaluation:

### **Question:** 
input_question

### **Reference Answer:** 
ref_answer

Here are the cropped subimages for you to evaluate:

---
'''

code_prm_suffix='''
#### Output Format (strictly follow)
Please provide an integer score to indicate the Validity of the cropped subimages. Output the score in a JSON dictionary with nothing else for easy processing, in this form: {"Validity": score}.

Your Evaluation Result:
'''

import re

import json
import re
import asyncio

async def evaluate_code_correctness(code_str, llm_openai_api, api_address, cur_port):
    """
    给定一段代码字符串，判断是否包含乱码或无意义内容。
    有则返回0，无则返回1，输出格式仅为：
    {"Correctness": 0} 或 {"Correctness": 1} 的JSON对象。

    参数：
    - code_str: str，待检测代码
    - llm_openai_api: 异步调用模型接口的函数
    - api_address: str，API地址
    - cur_port: str/int，API端口

    返回：
    - dict，例如 {"Correctness": 0} 或 {"Correctness": 1}
    """

    prompt = (
        "Please determine if the following code snippet contains any gibberish "
        "or meaningless content. Output 0 if it does, otherwise output 1. "
        "Only output a JSON dictionary in this exact format without any additional text: "
        "{\"Correctness\": 0 or 1}.\n\n"
        "Code:\n"
        f"{code_str}\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    max_try = 3
    for _ in range(max_try):
        try:
            completion = await llm_openai_api(messages, ip=api_address, host=cur_port)
            completion = completion[0] if isinstance(completion, list) else completion

            # 去除代码块 ``` 包裹内容
            if "```" in completion:
                pattern = re.compile(r'```(.*?)```', re.DOTALL)
                matches = pattern.findall(completion)
                if len(matches) == 1:
                    completion = matches[0]

            # 提取JSON字符串
            start_index = completion.find('{')
            end_index = completion.find('}', start_index + 1)
            if start_index == -1 or end_index == -1:
                raise ValueError("No JSON found in model output")

            json_str = completion[start_index:end_index+1]
            json_str = json_str.replace('\\', '\\\\')

            result = json.loads(json_str)
            correctness = int(result.get("Correctness", 0))
            return {"Correctness": correctness}

        except Exception:
            continue

    # 三次尝试失败，默认返回0
    return {"Correctness": 0}

def extract_thinking(solution):
    # 定义正则表达式模式
    thinking_pattern = r'<think>(.*?)</think>'
    
    # 使用正则表达式匹配
    thinking_match = re.search(thinking_pattern, solution, re.DOTALL)
    
    # 提取匹配结果，如果没有匹配到则返回空字符串
    thinking = thinking_match.group(1) if thinking_match else ''
    if thinking == "":
        thinking = solution.split("</think>")[0].replace("<think>", "")
    thinking = thinking.replace("<image>", "")
    thinking = thinking.replace("</sandbox_output>", "")
    thinking = thinking.replace("<sandbox_output>", "")
    return thinking

def extract_answer(solution):
    # 定义正则表达式模式
    answer_pattern = r'<answer>(.*?)</answer>'
    
    answer_match = re.search(answer_pattern, solution, re.DOTALL)
    
    answer = answer_match.group(1) if answer_match else ''
    # if answer == "":
    #     if "<answer>" in solution:
    #         answer = solution.split("<answer>")[-1].replace("</answer>", "")
    #     else:
    #         answer = solution[-100:]
    # answer = answer.replace("<image>", "")
    # answer = answer.replace("</sandbox_output>", "")
    # answer = answer.replace("<sandbox_output>", "")
    return answer.strip()

def format_check(solution):
    # 定义正则表达式，忽略空格、换行符等
    pattern = r"^(?:(?!</think>).)*</think>\s<answer>(?:(?!</answer>).)*</answer>\Z"
    # 使用re.fullmatch来检查整个字符串是否匹配
    match = re.fullmatch(pattern, solution, re.DOTALL)
    # with open("/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/examples/train/grpo/plugin/Format_Rewardcode_all_math_8.jsonl", "a+") as fout:
    #     fout.write(json.dumps({"solution": solution})+"\n")
    #     if match:
    #         fout.write(json.dumps({"match":match.group()})+"\n")
    #     else:
    #         fout.write(json.dumps({"match":match})+"\n")
    think_count = solution.count("<think>")
    think_end_count = solution.count("</think>")
    answer_count = solution.count("<answer>")
    answer_end_count = solution.count("</answer>")
    return match is not None  and think_count == 1 and answer_count == 1 and think_end_count == 1 and answer_end_count == 1

def code_check(solution):
    sandbox_pattern = r"<sandbox_output>(.*?)</sandbox_output>"
    box_matches = re.findall(sandbox_pattern, solution, re.DOTALL)
    correct_num = 0
    code_num = 0

    # <image>标签，允许连续多个，中间允许空白
    image_token_pattern = r"^(<image>\s*)+$"
    # 浮点数匹配，支持整数、小数、科学计数法
    float_pattern = r"^\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*$"
    for box_content in box_matches:
        code_num += 1
        content = box_content.strip()
        if not "sandbox for" in box_content.lower():
            correct_num += 1
    if code_num == 0:
        return False, 0
    return True, correct_num / code_num

def code_check_llm(solution):
    sandbox_pattern = r"<sandbox_output>(.*?)</sandbox_output>"
    box_matches = re.findall(sandbox_pattern, solution, re.DOTALL)
    correct_num = 0
    code_num = 0

    # <image>标签，允许连续多个，中间允许空白
    image_token_pattern = r"^(<image>\s*)+$"
    # 浮点数匹配，支持整数、小数、科学计数法
    float_pattern = r"^\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*$"
    for box_content in box_matches:
        code_num += 1
        content = box_content.strip()
        if not "sandbox for" in box_content.lower():
            correct_num += 1
    if code_num == 0:
        return False, 0
    return True, correct_num / code_num

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer: ",
        "Answer: ",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    return s

class VQAORM(ORM):
    def __init__(self):
        self.rank = int(os.getenv('RANK', -1))
        self.world_size = int(os.getenv('WORLD_SIZE', 1))
        self.api_address = os.getenv('REWARD_API_ADDRESS', 'http://localhost')
        self.weight = float(os.getenv('VQA_WEIGHT', 1))
        self.norm = bool(int(os.getenv('VQA_NORM', 1)))
        self.api_port_list = list(map(int, os.getenv('QWEN_API_PORT', '8080').split(',')))
        self.loop = asyncio.get_event_loop()
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        cur_task = self.loop.create_task(self.async_call(completions, solution, **kwargs))
        return self.loop.run_until_complete(cur_task)

    async def async_call(self, completions, solution, **kwargs) -> List[float]:
        # messages = kwargs.get('messages', [])
        images = kwargs.get('images', [])
        questions = kwargs.get('question', "")
        # image = images[0]
        # questions = [example[0]['content'] for example in messages]
        length = len(completions)
        tasks = []
        for cur_idx, (content, cur_solution, image_list, question) in enumerate(zip(completions, solution, images, questions)):
            cur_port = self.api_port_list[(self.rank * length + cur_idx) % len(self.api_port_list)]
            tasks.append(asyncio.create_task(self.evaluate(content, cur_solution, cur_port, None, question)))
        rewards = await asyncio.gather(*tasks)
        return rewards
    async def evaluate(self, cur_completion, cur_solution, cur_port, image, question) -> float:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        valid = False
        try:
            answer = extract_answer(cur_completion)
            if answer == '':
                return 0.
            
            student_answer = extract_characters_regex(answer)
            if '$\boxed' in student_answer:
                student_answer.replace('$\boxed', '$\\boxed')
            if evaluate_math(student_answer, cur_solution):
                return 1. * self.weight
            elif float(verify(parse(student_answer), parse(cur_solution))) > 0:
                return 1. * self.weight

            prompt = vqa_orm_template.replace("ref_answer", cur_solution).replace("gen_answer", answer).replace("input_question", question)
            # print(f"\nPrompt: {prompt}")
            # with open(image, "rb") as image_file:
            #     base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            messages = [
                {
                    "role": "user", 
                    "content": [
                        # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            max_try = 3
            outcome_r = 0
            completion = None
            for _ in range(max_try):
                try:
                    completion = await llm_openai_api(messages, ip=self.api_address, host=cur_port)
                    # completion = await get_model_prediction(messages)
                    completion = completion[0] if isinstance(completion, list) else completion
                    if "```" in completion:
                        pattern = re.compile(r'```(.*?)```', re.DOTALL)
                        matches = pattern.findall(completion)
                        if len(matches) == 1:
                            completion = matches[0]

                    json_str = completion[completion.find('{'): completion.find('}', completion.find('{') + 1) + 1]
                    json_str = json_str.replace('\\', '\\\\')
                    # if json_str == "":
                    #     print(f"\nEvaluation Result: {completion}")
                    completion = json.loads(json_str)
                    outcome_r = int(completion['Correctness'])
                    valid = True
                    break
                except timeout_decorator.TimeoutError:
                    print("Reward Calculation Timeout")
                    continue
                except Exception as e:
                    print(f"VQA_ORM: {e}, {completion}")
                    continue
        except Exception as e:
            print(e)
        if self.norm and valid:
            outcome_r = (outcome_r - 0.5) * 2
        if not valid:
            return INVALID_REWARD_VALUE

        # with open("./baseline_fmt_orm_cst_gen4_pixel2048_kl1e_3_lr1e_6_cst05_iterlimit8_ads_sample_800_gradnorm05.log.jsonl", "a+") as fout:
        #     fout.write(json.dumps({"cur_completion": cur_completion})+"\n")
        #     fout.write(json.dumps({"messages": prompt})+"\n")
        #     fout.write(json.dumps({"completion": completion})+"\n")
        #     fout.write(json.dumps({"outcome_r": outcome_r * self.weight})+"\n")

        return outcome_r * self.weight


class FMTORM(ORM):
    def __init__(self):
        self.weight = float(os.getenv('FMT_WEIGHT', 0.5))
        self.norm = bool(int(os.getenv('FMT_NORM', 0)))
        self.loop = asyncio.get_event_loop()
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        cur_task = self.loop.create_task(self.async_call(completions))
        return self.loop.run_until_complete(cur_task)

    async def async_call(self, completions) -> List[float]:
        tasks = []
        for content in completions:
            tasks.append(asyncio.create_task(self.evaluate(content)))
        rewards = await asyncio.gather(*tasks)
        return rewards

    async def evaluate(self, cur_completion) -> float:
        reward = 0
        if format_check(cur_completion):
            reward = self.weight
        elif self.norm:
            reward = -self.weight
        
        # with open("/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/examples/train/grpo/plugin/Format_Rewardcode_all_math_8.jsonl", "a+") as fout:
        #     fout.write(json.dumps({"reward": reward})+"\n")
        
        return reward


class CODEORM(ORM):
    def __init__(self):
        self.weight = float(os.getenv('CODE_WEIGHT', 0.3))
        self.loop = asyncio.get_event_loop()
        self.norm = bool(int(os.getenv('CODE_NORM', 0)))
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        cur_task = self.loop.create_task(self.async_call(completions))
        return self.loop.run_until_complete(cur_task)

    async def async_call(self, completions) -> List[float]:
        tasks = []
        for content in completions:
            tasks.append(asyncio.create_task(self.evaluate(content)))
        rewards = await asyncio.gather(*tasks)
        return rewards
    async def evaluate(self, cur_completion) -> float:
        answer = extract_answer(cur_completion)
        if answer == '':
            return 0.
        _, code_r = code_check(cur_completion)
        if self.norm:
            code_r = 2 * code_r - 1
        with open("/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/examples/train/grpo/plugin/Code_Reward.jsonl", "a+") as fout:
            fout.write(json.dumps({"cur_completion": cur_completion})+"\n")
            fout.write(json.dumps({"reward": code_r * self.weight})+"\n")
        
        return code_r * self.weight
async def evaluate_code_correctness(code_str, api_address, cur_port):
    """
    给定一段代码字符串，判断是否包含乱码或无意义内容。
    有则返回0，无则返回1，输出格式仅为：
    {"Correctness": 0} 或 {"Correctness": 1} 的JSON对象。

    参数：
    - code_str: str，待检测代码
    - llm_openai_api: 异步调用模型接口的函数
    - cur_port: str/int，API端口

    返回：
    - dict，例如 {"Correctness": 0} 或 {"Correctness": 1}
    """

    prompt = (
        "Please determine whether the following code snippet contains any gibberish or meaningless content. "
        "Alternatively, treat the code as useless if it consists entirely of comments without any image processing or computational logic. "
        "Output 0 if the code is meaningless or useless; otherwise, output 1. "
        "Respond only with a JSON dictionary in this exact format, without any extra text: "
        "{\"Correctness\": 0 or 1}.\n\n"
        "### Code:\n"
        f"{code_str}\n"
    )


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]
    max_try = 3
    for _ in range(max_try):
        try:
            completion = await llm_openai_api(messages, ip=api_address, host=cur_port)
            completion = completion[0] if isinstance(completion, list) else completion
            # 去除代码块 ``` 包裹内容
            if "```" in completion:
                pattern = re.compile(r'```(.*?)```', re.DOTALL)
                matches = pattern.findall(completion)
                if len(matches) == 1:
                    completion = matches[0]

            # 去除代码块 ``` 包裹内容
            if "```" in completion:
                pattern = re.compile(r'```(.*?)```', re.DOTALL)
                matches = pattern.findall(completion)
                if len(matches) == 1:
                    completion = matches[0]

            # 提取JSON字符串
            start_index = completion.find('{')
            end_index = completion.find('}', start_index + 1)
            if start_index == -1 or end_index == -1:
                raise ValueError("No JSON found in model output")

            json_str = completion[start_index:end_index+1]
            json_str = json_str.replace('\\', '\\\\')

            result = json.loads(json_str)
            correctness = int(result.get("Correctness", 0))
            return correctness
        except Exception:
            continue
    return 0

async def code_check_llm(solution, api_address, api_port):
    """
    最终修改版的函数。
    它查找成对的 `<code>` 和 `<sandbox_output>`。
    1. 它会忽略掉 <sandbox_output> 内容中包含 "sandbox for" 的代码块。
    2. 对于其他有效的代码块，它会提取 `<code>` 标签中的内容进行评估。
    """
    # 正则表达式使用两个捕获组，分别捕获 <code> 和 <sandbox_output> 内部的内容
    pattern = r"<code>(.*?)</code>.*?<sandbox_output>(.*?)</sandbox_output>"
    
    # re.findall 会返回一个元组列表，每个元组是 (code_content, sandbox_content)
    all_matches = re.findall(pattern, solution, re.DOTALL)
    
    # 如果没有找到任何匹配项，直接返回失败
    if not all_matches:
        return False, 0
    
    # 创建一个列表，只存放需要被评估的代码
    codes_to_evaluate = []
    for code_content, box_content in all_matches:
        # 核心检查：如果 sandbox_output 中不包含 "sandbox for"，则认为该代码块有效
        if "sandbox for" not in box_content.lower():
            codes_to_evaluate.append(code_content.strip())
        else:
            return False, 0

    # 如果过滤后没有需要评估的代码，也返回失败
    if not codes_to_evaluate:
        print("所有找到的代码块均被 'sandbox for' 规则过滤，没有可评估的代码。")
        return False, 0
        
    evaluated_count = 0
    for code in codes_to_evaluate:
        is_correct = await evaluate_code_correctness(code, api_address, api_port)
        # 只要有一段代码评估不正确，就立刻返回 False
        if not is_correct:
            return False, 0
        evaluated_count += 1

    # 如果所有被评估的代码块都正确，返回 True 和比率 (在这种逻辑下, 如果成功, 比率总是 1.0)
    return True, evaluated_count / len(codes_to_evaluate)


class CODEORMLLM(ORM):
    def __init__(self):
        self.weight = float(os.getenv('CODE_WEIGHT', 0.3))
        self.loop = asyncio.get_event_loop()
        self.norm = bool(int(os.getenv('CODE_NORM', 0)))

        self.rank = int(os.getenv('RANK', -1))
        self.world_size = int(os.getenv('WORLD_SIZE', 1))
        self.api_address = os.getenv('REWARD_API_ADDRESS', 'http://localhost')
        self.api_port_list = list(map(int, os.getenv('QWEN_API_PORT', '8080').split(',')))
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        cur_task = self.loop.create_task(self.async_call(completions))
        return self.loop.run_until_complete(cur_task)

    async def async_call(self, completions) -> List[float]:
        tasks = []
        for cur_idx, content in enumerate(completions):
            cur_port = self.api_port_list[(self.rank * len(completions) + cur_idx) % len(self.api_port_list)]
            tasks.append(asyncio.create_task(self.evaluate(content, cur_port)))
        rewards = await asyncio.gather(*tasks)
        return rewards
    async def evaluate(self, cur_completion, cur_port) -> float:
        answer = extract_answer(cur_completion)
        if answer == '':
            return 0.
        _, code_r = await code_check_llm(cur_completion, self.api_address, cur_port)
        if self.norm:
            code_r = 2 * code_r - 1
        with open("./examples/train/grpo/plugin/Code_Reward_llm_code.jsonl", "a+") as fout:
            fout.write(json.dumps({"cur_completion": cur_completion})+"\n")
            fout.write(json.dumps({"reward": code_r * self.weight})+"\n")
        
        return code_r * self.weight

class VQAPRM(ORM):
    def __init__(self):
        self.rank = int(os.getenv('RANK', -1))
        self.world_size = int(os.getenv('WORLD_SIZE', 1))
        self.api_address = os.getenv('REWARD_API_ADDRESS', 'http://localhost')
        self.weight = float(os.getenv('VQA_PRM_WEIGHT', 0.3))
        self.norm = bool(int(os.getenv('VQA_PRM_NORM', 0)))
        self.api_port_list = list(map(int, os.getenv('QWEN_API_PORT', '8080').split(',')))
        self.loop = asyncio.get_event_loop()
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        cur_task = self.loop.create_task(self.async_call(completions, solution, **kwargs))
        return self.loop.run_until_complete(cur_task)

    async def async_call(self, completions, solution, **kwargs) -> List[float]:
        # messages = kwargs.get('messages', [])
        images = kwargs.get('images', [])
        questions = kwargs.get('question', "")
        # image = images[0]
        # questions = [example[0]['content'] for example in messages]
        length = len(completions)
        tasks = []
        for cur_idx, (content, cur_solution, image_list, question) in enumerate(zip(completions, solution, images, questions)):
            cur_port = self.api_port_list[(self.rank * length + cur_idx) % len(self.api_port_list)]
            tasks.append(asyncio.create_task(self.evaluate(content, cur_solution, cur_port, None, question)))
        rewards = await asyncio.gather(*tasks)
        return rewards
    async def evaluate(self, cur_completion, cur_solution, cur_port, image, question) -> float:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        reward = 0.0
        valid = False
        try:
            thinking = extract_thinking(cur_completion)
            prompt = vqa_prm_template.replace("ref_answer", cur_solution).replace("gen_solution", thinking).replace("input_question", question)
            # print(f"\nPrompt: {prompt}")
            # with open(image, "rb") as image_file:
            #     base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            messages = [
                {
                    "role": "user", 
                    "content": [
                        # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            max_try = 3
            r = 0
            completion = None
            for _ in range(max_try):
                try:
                    r = 0
                    completion = await llm_openai_api(messages, ip=self.api_address, host=cur_port)
                    # completion = await get_model_prediction(messages)
                    completion = completion[0] if isinstance(completion, list) else completion
                    if "```" in completion:
                        pattern = re.compile(r'```(.*?)```', re.DOTALL)
                        matches = pattern.findall(completion)
                        if len(matches) == 1:
                            completion = matches[0]

                    json_str = completion[completion.find('{'): completion.find('}', completion.find('{') + 1) + 1]
                    json_str = json_str.replace('\\', '\\\\')
                    # if json_str == "":
                    #     print(f"\nEvaluation Result: {completion}")
                    completion = json.loads(json_str)
                    process_c = int(completion['Process Correctness']) / 10
                    text_q = int(completion['Text Quality']) / 10
                    r += process_c * 0.5
                    r += text_q * 0.5
                    valid = True
                    break
                except timeout_decorator.TimeoutError:
                    print("Reward Calculation Timeout")
                    continue
                except Exception as e:
                    print(f"VQA_PRM: {e}, {completion}")
                    continue
            reward += r
        except Exception as e:
            print(e)
        if self.norm and valid:
            reward = (reward - 0.5) * 2
        if not valid:
            return INVALID_REWARD_VALUE
        # messages = [for message in messages]
        # with open("/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/examples/train/grpo/plugin/VQA_Process_Rewardcode_all_math_8.jsonl", "a+") as fout:
        #     fout.write(json.dumps({"cur_completion": cur_completion})+"\n")
        #     fout.write(json.dumps({"messages": prompt})+"\n")
        #     fout.write(json.dumps({"completion": completion})+"\n")
        #     fout.write(json.dumps({"outcome_r": reward * self.weight})+"\n")
        # print(f"\nReward: {reward}")
        return reward * self.weight

class CODEPRM(ORM):
    def __init__(self):
        self.rank = int(os.getenv('RANK', -1))
        self.world_size = int(os.getenv('WORLD_SIZE', 1))
        self.api_address = os.getenv('REWARD_API_ADDRESS', 'http://localhost')
        self.weight = float(os.getenv('CODE_PRM_WEIGHT', 0.3))
        self.api_port_list = list(map(int, os.getenv('QWEN_API_PORT', '8080').split(',')))
        self.loop = asyncio.get_event_loop()
        self.norm = bool(int(os.getenv('CODE_PRM_NORM', 0)))
        self.avg = bool(int(os.getenv('CODE_PRM_AVG', 0)))
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        cur_task = self.loop.create_task(self.async_call(completions, solution, **kwargs))
        return self.loop.run_until_complete(cur_task)

    async def async_call(self, completions, solution, **kwargs) -> List[float]:
        # messages = kwargs.get('messages', [])
        images = kwargs.get('images', [])
        questions = kwargs.get('question', "")
        # image = images[0]
        # questions = [example[0]['content'] for example in messages]
        length = len(completions)
        tasks = []
        for cur_idx, (content, cur_solution, image_list, question) in enumerate(zip(completions, solution, images, questions)):
            image_paths = [image["path"] for image in image_list]
            cur_port = self.api_port_list[(self.rank * length + cur_idx) % len(self.api_port_list)]
            tasks.append(asyncio.create_task(self.evaluate(content, cur_solution, cur_port, image_paths, question)))
        rewards = await asyncio.gather(*tasks)
        return rewards

    async def evaluate(self, cur_completion, cur_solution, cur_port, image_paths, question) -> float:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        valid = False
        reward = 0.0
        if len(image_paths) != 1:
            try:
                ori_image = image_paths[0]
                sub_image_paths = image_paths[1:]
                codes = cur_completion.split("</sandbox_output>")
                prompt = code_prm_template.replace("ref_answer", cur_solution).replace("input_question", question)
                with open(ori_image, "rb") as image_file:
                    ori_base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                base_messages = {
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ori_base64_image}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
                messages = []
                # print(f"\nPrompt: {prompt}")
                begin_idx = 0
                msg_imgs = []
                for code in codes:
                    if begin_idx >= len(sub_image_paths):
                        break
                    messages.append(base_messages)
                    # sandbox_pattern = r"<sandbox_output>(.*?)</sandbox_output>"
                    # box_matches = re.search(sandbox_pattern, cur_solution, re.DOTALL)
                    # box_content = box_matches.group(1) if box_matches else ''
                    image_count = code.count("<image>")
                    if image_count == 0:
                        continue
                    msg_imgs.append([image_count, []])
                    for sub_idx in range(image_count):
                        if begin_idx + sub_idx >= len(sub_image_paths):
                            break
                        sub_image_path = sub_image_paths[begin_idx + sub_idx]
                        with open(sub_image_path, "rb") as image_file:
                            sub_base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                        messages[-1]["content"].append(
                            {"type": "text", "text": f"Sub Image {sub_idx}: "}
                        )
                        messages[-1]["content"].append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{sub_base64_image}"}},
                        )
                        msg_imgs[-1][-1].append(sub_image_path)
                    messages[-1]["content"].append(
                        {"type": "text", "text": code_prm_suffix}
                    )
                    begin_idx += image_count
                    
                max_try = 3
                r = 0
                valid = False
                completions = []
                loaded_completions = []
                for _ in range(max_try):
                    try:
                        r = 0
                        for message in messages:
                            completion = await llm_openai_api([message], ip=self.api_address, host=cur_port)
                            # completion = await get_model_prediction([message])
                            completion = completion[0] if isinstance(completion, list) else completion
                            completions.append(completion)
                        for completion in completions:
                            completion = completion[0] if isinstance(completion, list) else completion
                            if "```" in completion:
                                pattern = re.compile(r'```(.*?)```', re.DOTALL)
                                matches = pattern.findall(completion)
                                if len(matches) == 1:
                                    completion = matches[0]

                            json_str = completion[completion.find('{'): completion.find('}', completion.find('{') + 1) + 1]
                            json_str = json_str.replace('\\', '\\\\')
                            # if json_str == "":
                            #     print(f"\nEvaluation Result: {completion}")
                            completion = json.loads(json_str)
                            validity = int(completion['Validity'])
                            r += validity
                            valid = True
                            loaded_completions.append(completion)
                        break
                    except timeout_decorator.TimeoutError:
                        print("Reward Calculation Timeout")
                        continue
                    except Exception as e:
                        print(f"CODE_PRM: {e}, {completions}")
                        continue
                if self.avg:
                    reward = r / len(messages)
                elif r > 0:
                    reward = 1
            except Exception as e:
                print(e)
        else: # No sub image to check
            valid = True
            if self.norm:
                reward = 0.5
            else:
                reward = 0
        if self.norm and valid:
            # if not valid:
            #     reward = 0.49
            reward = (reward - 0.5) * 2
        # print(f"\nReward: {reward}")
        
        if not valid:
            return INVALID_REWARD_VALUE
        # with open("/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/examples/train/grpo/plugin/Code_Process_Rewardcode_all_math_8.jsonl", "a+") as fout:
        #     fout.write(json.dumps({"cur_completion": cur_completion})+"\n")
        #     if len(image_paths) != 1:
        #         for message in messages:
        #             for content in message["content"]:
        #                 if content["type"] == "image_url":
        #                     content["image_url"] = "img"
        #         fout.write(json.dumps({"messages": messages})+"\n")
        #         fout.write(json.dumps({"completions": completions})+"\n")
        #         fout.write(json.dumps({"codes": codes})+"\n")
        #         fout.write(json.dumps({"round_images": msg_imgs})+"\n")
        #         fout.write(json.dumps({"sub_image_paths": sub_image_paths})+"\n")
        #     fout.write(json.dumps({"reward": reward * self.weight})+"\n")
        return reward * self.weight


consistency_template = '''
You are an expert about question answering. I will provide you a solution process and a final answer to the same question. Please evaluate the Consistency between the solution process and the final anwer: If the solution process draws the same conclusion with the final answer, rate the Consistency as 1, else rate 0.
Here is the solution process and the final answer for you to evaluate:\n

#### Solution Process: 
gen_solution\n

#### Final Answer: 
gen_answer\n

### Output Format (strictly follow)

Please provide an integer score to indicate the Consistency. Output the score in a JSON dictionary with nothing else for easy processing, in this form: {"Consistency": score}.

Your Evaluation Result:
'''

class CSTORM(ORM):
    def __init__(self):
        self.rank = int(os.getenv('RANK', -1))
        self.world_size = int(os.getenv('WORLD_SIZE', 1))
        self.api_address = os.getenv('REWARD_API_ADDRESS', 'http://localhost')
        self.weight = float(os.getenv('CST_WEIGHT', 0.5))
        self.norm = bool(int(os.getenv('CST_NORM', 0)))
        self.api_port_list = list(map(int, os.getenv('QWEN_API_PORT', '8080').split(',')))
        self.loop = asyncio.get_event_loop()
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        cur_task = self.loop.create_task(self.async_call(completions, solution, **kwargs))
        return self.loop.run_until_complete(cur_task)

    async def async_call(self, completions, solution, **kwargs) -> List[float]:
        # messages = kwargs.get('messages', [])
        images = kwargs.get('images', [])
        questions = kwargs.get('question', "")
        # image = images[0]
        # questions = [example[0]['content'] for example in messages]
        length = len(completions)
        tasks = []
        for cur_idx, (content, cur_solution, image_list, question) in enumerate(zip(completions, solution, images, questions)):
            cur_port = self.api_port_list[(self.rank * length + cur_idx) % len(self.api_port_list)]
            tasks.append(asyncio.create_task(self.evaluate(content, cur_solution, cur_port, None, question)))
        rewards = await asyncio.gather(*tasks)
        return rewards
    async def evaluate(self, cur_completion, cur_solution, cur_port, image, question) -> float:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        valid = False
        try:
            thinking = extract_thinking(cur_completion)[-500:]
            answer = extract_answer(cur_completion)
            if answer == '':
                return 0.
            prompt = consistency_template.replace("gen_solution", thinking).replace("gen_answer", answer)
            # print(f"\nPrompt: {prompt}")
            # with open(image, "rb") as image_file:
            #     base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            max_try = 3
            outcome_r = 0
            completion = None
            for _ in range(max_try):
                try:
                    completion = await llm_openai_api(messages, ip=self.api_address, host=cur_port)
                    # completion = await get_model_prediction(messages)
                    completion = completion[0] if isinstance(completion, list) else completion
                    if "```" in completion:
                        pattern = re.compile(r'```(.*?)```', re.DOTALL)
                        matches = pattern.findall(completion)
                        if len(matches) == 1:
                            completion = matches[0]

                    json_str = completion[completion.find('{'): completion.find('}', completion.find('{') + 1) + 1]
                    json_str = json_str.replace('\\', '\\\\')
                    # if json_str == "":
                    #     print(f"\nEvaluation Result: {completion}")
                    completion = json.loads(json_str)
                    outcome_r = int(completion['Consistency'])
                    valid = True
                    break
                except timeout_decorator.TimeoutError:
                    print("Reward Calculation Timeout")
                    continue
                except Exception as e:
                    print(f"CST_ORM: {e}, {completion}")
                    continue
        except Exception as e:
            print(e)
        if self.norm and valid:
            # if not valid:
            #     outcome_r = 0.49
            outcome_r = (outcome_r - 0.5) * 2
        if not valid:
            return INVALID_REWARD_VALUE
        # if random.random() < 0.1:
        # with open("/mllm_hdd/yfzhang/Agent-R1/agent_latest_code/examples/train/grpo/plugin/CST_Rewardcode_all_math_8.jsonl", "a+") as fout:
        #     fout.write(json.dumps({"cur_completion": cur_completion})+"\n")
        #     fout.write(json.dumps({"messages": prompt})+"\n")
        #     fout.write(json.dumps({"completion": completion})+"\n")
        #     fout.write(json.dumps({"outcome_r": outcome_r * self.weight})+"\n")

        return outcome_r * self.weight


orms['vqa_orm'] = VQAORM
orms['fmt_orm'] = FMTORM
orms['code_orm'] = CODEORM
orms['code_orm_llm'] = CODEORMLLM
orms['vqa_prm'] = VQAPRM
orms['code_prm'] = CODEPRM
orms['cst_orm'] = CSTORM

