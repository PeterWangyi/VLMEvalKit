# flake8: noqa
import ast
import os.path as osp
import decord
import re
import math

from ..smp import *
from ..smp.file import LMUDataRoot, load, getenv_bool
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE


class SPBench(ImageBaseDataset):
    TYPE = 'MCQ'

    THINKING_TEMPLATE = (
        "Question: {question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. \n"
    )

    PROMPT_TEMPLATES = {
        "default":
        {
            "pre_prompt": "Question: {question}\n",
            "mca_post_prompt": "Please answer with the option's letter from the given choices (e.g., A, B, etc.) directly.",
            "na_post_prompt": "Please answer the question using a numerical value (e.g., 42 or 3.1) directly."
        },
        "thinking":
        {
            "pre_prompt": THINKING_TEMPLATE,
            "mca_post_prompt": (
                "Please provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question with the option's letter from the given choices (e.g., A, B, etc.) within the <answer> </answer> tags."
            ),
            "na_post_prompt": (
                "Please provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question with a numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags."
            )
        },
    }

    SPBENCH_TASKS = [
        'MV',
        'SI'
    ]

    LMUData_root = LMUDataRoot()
    DATASET_URL = {}

    for task in SPBENCH_TASKS:
        name = f"SPBench-{task}"
        path = osp.join(LMUData_root, name + ".tsv")
        DATASET_URL[name] = path

    DATASET_MD5 = {key: None for key in DATASET_URL}

    @classmethod
    def get_task_type(self, task):
        MCA_QUESTION_TYPES = [
            "object_rel_direction",
            "object_rel_distance"
        ]
        NA_QUESTION_TYPES = [
            "object_counting",
            "object_size_estimation",
            "object_abs_distance",
            "object_size_estimation"
        ]

        if task in MCA_QUESTION_TYPES:
            return 'MCQ'
        elif task in NA_QUESTION_TYPES:
            return 'NA'
        else:
            raise ValueError('')

    def build_prompt(self, line):

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = line['candidates']

        if options is None:
            options = []
        elif isinstance(options, str):
            try:
                options = ast.literal_eval(options)  # 字符串 -> 列表
            except Exception:
                options = [options] if options.strip() else []
        elif not isinstance(options, (list, tuple)):
            options = [options]

        if options:
            question += "\nOptions:\n" + "\n".join(options)

        question_type = line['question_type']
        task_type = self.get_task_type(question_type)

        # default prompt type in SRbench
        prompt_type = 'thinking'
        prompt_template = self.PROMPT_TEMPLATES.get(prompt_type)

        prompt_text = prompt_template["pre_prompt"].format(question=question)
        if task_type == 'MCQ':
            prompt_text += "\n" + prompt_template['mca_post_prompt']
        elif task_type == 'NA':
            prompt_text += "\n" + prompt_template["na_post_prompt"]

        print(f"prompt: {prompt_text}")

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt_text))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        # call the base class evaluate method, because we want to use the official matching strategy.
        # The resaon why we override this method is that we print the results in base calss evaluate method,
        # and we want to return None here to avoid printing the results again.
        # super().evaluate(eval_file, **judge_kwargs)

        # nproc = judge_kwargs.pop('nproc', 4)

        # suffix = eval_file.split('.')[-1]
        # model = judge_kwargs.get('model', 'exact_matching')
        # assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125', 'gpt-4-turbo-proxy', 'gpt-4o-2024-11-20_proxy']
        # name_str_map = {
        #     'chatgpt-0125': 'openai',
        #     'gpt-4-0125': 'gpt4',
        #     'gpt-4-turbo-proxy': 'gpt4-turbo',
        #     'gpt-4o-2024-11-20_proxy': 'gpt-4o-2024-11-20'
        # }

        return None
