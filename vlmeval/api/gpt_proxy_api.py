import os
import numpy as np
from .base import BaseAPI
import openai_proxy
from PIL import Image
from .gpt import encode_image_to_base64


class OpenAIProxyWrapper(BaseAPI):
    is_api: bool = True

    def __init__(self,
                 model: str = "gpt-4o-2024-08-06",
                 retry: int = 3,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 300,
                 max_tokens: int = 2048,
                 img_size: int = 512,
                 img_detail: str = 'low',
                 **kwargs):

        self.model = model
        self.o1_model = False

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.img_size = img_size
        self.img_detail = img_detail
        self.timeout = timeout
        self.fail_msg = 'Failed to obtain answer via GPT proxy. '

        self.key = os.getenv("GPT_PROXY_KEY", key)
        assert self.key is not None, 'Please set the environment variable GPT_PROXY_KEY.'

        self.client = openai_proxy.GptProxy(api_key=self.key)
        self.transaction_prefix = f"vlmevalkit_{self.model}_v2"

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs):
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        transaction_id = f"{self.transaction_prefix}"
        channel_code = "doubao" if "doubao" in self.model.lower() else None

        payload = dict(
            messages=input_msgs,
            model=self.model,
            transaction_id=transaction_id,
            temperature=temperature,
            channel_code=channel_code,
            **kwargs)

        if self.o1_model:
            reasoning_effort = os.getenv('gpt5_reasoning_effort', 'medium')
            gpt5_max_tokens = int(os.getenv('gpt5_max_tokens', 2048))

            payload['max_completion_tokens'] = gpt5_max_tokens
            payload['reasoning_effort'] = reasoning_effort

            payload.pop('temperature')
        else:
            payload['max_tokens'] = max_tokens

        rsp = self.client.generate(
            **payload
        )

        ret_code = 0 if rsp.ok else -1
        try:
            content_str = rsp.json()["data"]["response_content"]["choices"][0]["message"]["content"]
            answer = content_str.strip()
        except Exception:
            answer = self.fail_msg
        return ret_code, answer, rsp


class GPT4VProxy(OpenAIProxyWrapper):
    def generate(self, message, dataset=None):
        return super(GPT4VProxy, self).generate(message)


class GPT5VProxy(OpenAIProxyWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.o1_model = True

    def generate(self, message, dataset=None):
        index = None
        if isinstance(message, list):
            new_msgs = []
            for item in message:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "index"
                    and index is None
                ):
                    index = str(item.get("value"))
                else:
                    new_msgs.append(item)
            message = new_msgs

        return super().generate(message, index=index)
