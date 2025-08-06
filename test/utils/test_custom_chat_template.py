import os
import unittest
from fastdeploy.entrypoints.chat_utils import load_chat_template
from fastdeploy.input.ernie_processor import ErnieProcessor
from fastdeploy.input.text_processor import DataProcessor
from fastdeploy.input.ernie_vl_processor import ErnieMoEVLProcessor

input_chat_template = "unit test \n"

class TestChatTemplate(unittest.TestCase):
    def test_load_chat_template_str(self):
        result = load_chat_template(input_chat_template)
        self.assertEqual(input_chat_template, result)

    def test_load_chat_template_path(self):
        with open("chat_template", 'w', encoding='utf-8') as file:
            file.write(input_chat_template)
        file_path = os.path.join(os.getcwd(), "chat_template")
        result = load_chat_template(file_path)
        self.assertEqual(input_chat_template, result)

    def test_apply_chat_template_ernie(self):
        base_path = os.getenv("MODEL_PATH")
        if base_path:
            model_path = os.path.join(base_path, "ernie-4_5-21b-a3b-bf16-paddle")
        else:
            model_path = "./ernie-4_5-21b-a3b-bf16-paddle"
        request = {
            "messages":[
                {"role": "user", "content": "Hello"}
            ],
            "chat_template": input_chat_template
        }
        processor = ErnieProcessor(model_path)
        ids = processor.messages2ids(request)
        result = processor.tokenizer.decode(ids)
        self.assertEqual(input_chat_template, result)

    def test_apply_chat_template_text(self):
        base_path = os.getenv("MODEL_PATH")
        if base_path:
            model_path = os.path.join(base_path, "Qwen2-7B-Instruct")
        else:
            model_path = "./Qwen2-7B-Instruct"
        request = {
            "messages":[
                {"role": "user", "content": "Hello"}
            ],
            "chat_template": input_chat_template
        }
        processor = DataProcessor(model_path)
        ids = processor.messages2ids(request)
        result = processor.tokenizer.decode(ids)
        self.assertEqual(input_chat_template, result)

    def test_apply_chat_template_vl(self):
        base_path = os.getenv("MODEL_PATH")
        if base_path:
            model_path = os.path.join(base_path, "ernie-4_5-vl-28b-a3b-bf16-paddle")
        else:
            model_path = "./ernie-4_5-vl-28b-a3b-bf16-paddle"
        request = {
            "messages":[
                {"role": "user", "content": "Hello"}
            ],
            "chat_template": input_chat_template
        }
        processor = ErnieMoEVLProcessor(model_path)
        ids = processor.ernie_processor.apply_chat_template(request)
        result = processor.tokenizer.decode(ids)
        self.assertEqual(input_chat_template, result)


if __name__ == "__main__":
    unittest.main()