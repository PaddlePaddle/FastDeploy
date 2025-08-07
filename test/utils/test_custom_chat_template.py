import os
import unittest
from fastdeploy.entrypoints.chat_utils import load_chat_template

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
        os.remove(file_path)
        self.assertEqual(input_chat_template, result)

if __name__ == "__main__":
    unittest.main()
    