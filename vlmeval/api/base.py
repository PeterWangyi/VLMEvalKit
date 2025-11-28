import time
import random as rd
import re
import json
import copy as cp
import base64

from abc import abstractmethod
from ..smp import get_logger, parse_file, concat_images_vlmeval, LMUDataRoot, md5, decode_base64_to_image_file
from json import JSONDecodeError


class BaseAPI:

    allowed_types = ['text', 'image', 'video']
    INTERLEAVE = True
    INSTALL_REQ = False

    def __init__(self,
                 retry=10,
                 wait=1,
                 system_prompt=None,
                 verbose=True,
                 fail_msg='Failed to obtain answer via API.',
                 **kwargs):
        """Base Class for all APIs.

        Args:
            retry (int, optional): The retry times for `generate_inner`. Defaults to 10.
            wait (int, optional): The wait time after each failed retry of `generate_inner`. Defaults to 1.
            system_prompt (str, optional): Defaults to None.
            verbose (bool, optional): Defaults to True.
            fail_msg (str, optional): The message to return when failed to obtain answer.
                Defaults to 'Failed to obtain answer via API.'.
            **kwargs: Other kwargs for `generate_inner`.
        """

        self.wait = wait
        self.retry = retry
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.fail_msg = fail_msg
        self.logger = get_logger('ChatAPI')

        self.length_msg = 'Hit max new token.'

        if len(kwargs):
            self.logger.info(f'BaseAPI received the following kwargs: {kwargs}')
            self.logger.info('Will try to use them as kwargs for `generate`. ')
        self.default_kwargs = kwargs

    @abstractmethod
    def generate_inner(self, inputs, **kwargs):
        """The inner function to generate the answer.

        Returns:
            tuple(int, str, str): ret_code, response, log
        """
        self.logger.warning('For APIBase, generate_inner is an abstract method. ')
        assert 0, 'generate_inner not defined'
        ret_code, answer, log = None, None, None
        # if ret_code is 0, means succeed
        return ret_code, answer, log

    def working(self):
        """If the API model is working, return True, else return False.

        Returns:
            bool: If the API model is working, return True, else return False.
        """
        self.old_timeout = None
        if hasattr(self, 'timeout'):
            self.old_timeout = self.timeout
            self.timeout = 120

        retry = 5
        while retry > 0:
            ret = self.generate('hello')
            if ret is not None and ret != '' and self.fail_msg not in ret:
                if self.old_timeout is not None:
                    self.timeout = self.old_timeout
                return True
            retry -= 1

        if self.old_timeout is not None:
            self.timeout = self.old_timeout
        return False

    def check_content(self, msgs):
        """Check the content type of the input. Four types are allowed: str, dict, liststr, listdict.

        Args:
            msgs: Raw input messages.

        Returns:
            str: The message type.
        """
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'

    def preproc_content(self, inputs):
        """Convert the raw input messages to a list of dicts.

        Args:
            inputs: raw input messages.

        Returns:
            list(dict): The preprocessed input messages. Will return None if failed to preprocess the input.
        """
        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif self.check_content(inputs) == 'dict':
            assert 'type' in inputs and 'value' in inputs
            return [inputs]
        elif self.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                mime, pth = parse_file(s)
                if mime is None or mime == 'unknown':
                    res.append(dict(type='text', value=s))
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))
            return res
        elif self.check_content(inputs) == 'listdict':
            for item in inputs:
                assert 'type' in item and 'value' in item
                mime, s = parse_file(item['value'])
                if mime is None:
                    assert item['type'] == 'text', item['value']
                else:
                    assert mime.split('/')[0] == item['type']
                    item['value'] = s
            return inputs
        else:
            return None

    # May exceed the context windows size, so try with different turn numbers.
    def chat_inner(self, inputs, **kwargs):
        _ = kwargs.pop('dataset', None)
        while len(inputs):
            try:
                return self.generate_inner(inputs, **kwargs)
            except Exception as e:
                if self.verbose:
                    self.logger.info(f'{type(e)}: {e}')
                inputs = inputs[1:]
                while len(inputs) and inputs[0]['role'] != 'user':
                    inputs = inputs[1:]
                continue
        return -1, self.fail_msg + ': ' + 'Failed with all possible conversation turns.', None

    def chat(self, messages, **kwargs1):
        """The main function for multi-turn chatting. Will call `chat_inner` with the preprocessed input messages."""
        assert hasattr(self, 'chat_inner'), 'The API model should has the `chat_inner` method. '
        for msg in messages:
            assert isinstance(msg, dict) and 'role' in msg and 'content' in msg, msg
            assert self.check_content(msg['content']) in ['str', 'dict', 'liststr', 'listdict'], msg
            msg['content'] = self.preproc_content(msg['content'])
        # merge kwargs
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)

        answer = None
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        time.sleep(T)

        assert messages[-1]['role'] == 'user'

        for i in range(self.retry):
            try:
                ret_code, answer, log = self.chat_inner(messages, **kwargs)
                if ret_code == 0 and self.fail_msg not in answer and answer != '':
                    if self.verbose:
                        print(answer)
                    return answer
                elif self.verbose:
                    if not isinstance(log, str):
                        try:
                            log = log.text
                        except Exception as e:
                            self.logger.warning(f'Failed to parse {log} as an http response: {str(e)}. ')
                    self.logger.info(f'RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}')
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'An error occured during try {i}: ')
                    self.logger.error(f'{type(err)}: {err}')
            # delay before each retry
            T = rd.random() * self.wait * 2
            time.sleep(T)

        return self.fail_msg if answer in ['', None] else answer

    def preprocess_message_with_role(self, message):
        system_prompt = ''
        new_message = []

        for data in message:
            assert isinstance(data, dict)
            role = data.pop('role', 'user')
            if role == 'system':
                system_prompt += data['value'] + '\n'
            else:
                new_message.append(data)

        if system_prompt != '':
            if self.system_prompt is None:
                self.system_prompt = system_prompt
            else:
                if system_prompt not in self.system_prompt:
                    self.system_prompt += '\n' + system_prompt
        return new_message

    def generate(self, message, **kwargs1):
        """The main function to generate the answer. Will call `generate_inner` with the preprocessed input messages.

        Args:
            message: raw input messages.

        Returns:
            str: The generated answer of the Failed Message if failed to obtain answer.
        """

        index = kwargs1.pop("index", None)

        if self.check_content(message) == 'listdict':
            message = self.preprocess_message_with_role(message)

        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        message = self.preproc_content(message)
        assert message is not None and self.check_content(message) == 'listdict'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'

        # merge kwargs
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)

        answer = None
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        time.sleep(T)

        for i in range(self.retry):
            try:
                ret_code, answer, log = self.generate_inner(message, **kwargs)
                log_preview = preview_response(log)

                # 用于字段提取的 dict
                try:
                    from requests import Response
                    if isinstance(log, Response):
                        try:
                            log_json = log.json()
                        except Exception:
                            # 有些返回 header 没标 JSON，但 body 是 JSON
                            try:
                                log_json = json.loads(getattr(log, "text", "") or "")
                            except Exception:
                                log_json = {}
                    elif isinstance(log, dict):
                        log_json = log
                    elif isinstance(log, str):
                        try:
                            log_json = json.loads(log)
                        except Exception:
                            log_json = {}
                    else:
                        log_json = {}
                except Exception:
                    log_json = {}

                # # 兼容多 schema 提取 finish_reason
                # choices = (
                #     (log_json.get("data", {}).get("response_content", {}) or {}).get("choices")
                #     or log_json.get("choices")
                #     or []
                # )
                # finish_reason = (
                #     choices[0].get("finish_reason")
                #     if isinstance(choices, list) and choices and isinstance(choices[0], dict)
                #     else None
                # )
                finish_reason = get_finish_reason(log_json)

                # 正常拿到答案
                # if ret_code == 0 and self.fail_msg not in answer and answer != '':
                if ret_code == 0 and self.fail_msg not in answer:
                    if self.verbose:
                        self.logger.info(
                            f"[index={index}], RetCode: {ret_code}, finish reason: {finish_reason}\n"
                            f"Answer: {answer}\n"
                            f"Log{index}: {log_preview}"
                        )

                    is_length = finish_reason in ("length", "MAX_TOKENS")
                    return self.length_msg if is_length else answer
                    # return self.length_msg if finish_reason == 'length' else answer
                elif self.verbose:
                    if not isinstance(log, str):
                        try:
                            log = log.text
                        except Exception as e:
                            self.logger.warning(f'Failed to parse {log} as an http response: {str(e)}. ')
                    self.logger.info(
                        f"[index={index}], RetCode: {ret_code}\n"
                        f"QuestionAnswer: {answer}\n"
                        f"Log{index}: {log_preview}"
                    )
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'An error occured during try {i}: ')
                    self.logger.error(f'{type(err)}: {err}')
            # delay before each retry
            T = rd.random() * self.wait * 2
            time.sleep(T)

        return self.fail_msg if answer in ['', None] else answer

    def message_to_promptimg(self, message, dataset=None):
        assert not self.INTERLEAVE
        model_name = self.__class__.__name__
        import warnings
        warnings.warn(
            f'Model {model_name} does not support interleaved input. '
            'Will use the first image and aggregated texts as prompt. ')
        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = None
        elif num_images == 1:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [x['value'] for x in message if x['type'] == 'image'][0]
        else:
            prompt = '\n'.join([x['value'] if x['type'] == 'text' else '<image>' for x in message])
            if dataset == 'BLINK':
                image = concat_images_vlmeval(
                    [x['value'] for x in message if x['type'] == 'image'],
                    target_size=512)
            else:
                image = [x['value'] for x in message if x['type'] == 'image'][0]
        return prompt, image

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func


_DATA_URL_RE = re.compile(
    r'data:([-\w.+/]+);base64,([A-Za-z0-9+/=]{400,})',  # 100+ 视为“大段”
    re.IGNORECASE
)


def _redact_data_urls(s: str) -> str:
    def _repl(m):
        mime = m.group(1)
        n = len(m.group(2))
        return f"<data-url:{mime}; base64 {n} chars omitted>"
    return _DATA_URL_RE.sub(_repl, s)


def preview_response(resp, max_len=20000):
    try:
        from requests import Response
        if isinstance(resp, Response):
            ct = (resp.headers.get("content-type") or "").lower()
            status = resp.status_code

            if "application/json" in ct:
                try:
                    d = resp.json()
                    s = json.dumps(d, ensure_ascii=False)
                except JSONDecodeError:
                    s = resp.text
                # 只在这里做 base64 折叠
                s = _redact_data_urls(s)
                return f"HTTP {status} | {s[:max_len]}"

            # 非 JSON：直接 text，且只折叠 data URL
            s = resp.text
            s = _redact_data_urls(s)
            return f"HTTP {status} | {s[:max_len]}"
    except Exception:
        pass

    # 非 Response：dict/list/str；也只折叠 data URL
    if isinstance(resp, (dict, list)):
        s = json.dumps(resp, ensure_ascii=False)
        s = _redact_data_urls(s)
        return s[:max_len]

    s = str(resp)
    s = _redact_data_urls(s)
    return s[:max_len]


def response_to_dict(resp):
    from requests import Response
    if isinstance(resp, Response):
        try:
            return resp.json()  # 优先尝试 JSON
        except Exception:
            ct = resp.headers.get("content-type", "")
            if "text" in ct or "html" in ct:
                return {"raw_text": resp.text}
            else:
                return {"raw_bytes": base64.b64encode(resp.content).decode()}
    elif isinstance(resp, str):
        try:
            return json.loads(resp)
        except Exception:
            return {"raw_text": resp}
    elif isinstance(resp, dict):
        return resp
    else:
        return {"raw": str(resp)}


def get_finish_reason(log_json):
    """只提取 finish_reason；若没有则尝试 native_finish_reason。绝不做标准化。"""
    if not isinstance(log_json, dict):
        return None

    # 按优先级尝试几条最常见路径
    paths = [
        ("data", "response_content", "choices"),  # 你的旧情况
        ("choices",),                             # 你的新例子/常见
        ("data", "choices"),                      # 偶尔有厂商用这个
    ]

    for path in paths:
        node = log_json
        for k in path:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                node = None
                break
        if node is None:
            continue

        # 取第一个 choice
        if isinstance(node, list) and node:
            first = node[0]
        elif isinstance(node, dict):
            first = node
        else:
            continue

        if isinstance(first, dict):
            # 直接拿 finish_reason
            fr = first.get("finish_reason")
            if fr is not None:
                return fr
            # 没有就兜底拿 native_finish_reason
            nfr = first.get("native_finish_reason")
            if nfr is not None:
                return nfr
            # 某些在 message 里
            msg = first.get("message")
            if isinstance(msg, dict):
                fr = msg.get("finish_reason")
                if fr is not None:
                    return fr
                nfr = msg.get("native_finish_reason")
                if nfr is not None:
                    return nfr

    return None
