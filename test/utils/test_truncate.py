import unittest

from fastdeploy.utils import truncate_text


class TestTruncateText(unittest.TestCase):
    def test_truncate_prompt(self):
        data = {"prompt": "a" * 20000}
        result = truncate_text(data)
        self.assertTrue("..." in result["prompt"])

    def test_truncate_messages(self):
        data = {"messages": [{"content": "short"}, {"content": "long" * 10000}]}
        result = truncate_text(data)
        self.assertEqual(len(result["messages"][0]["content"]), 5)
        self.assertTrue(len(result["messages"][1]["content"]) < 40000)
        self.assertTrue("..." in result["messages"][1]["content"])

    def test_no_truncate_needed(self):
        data = {"prompt": "short"}
        result = truncate_text(data)
        self.assertEqual(result["prompt"], "short")


if __name__ == "__main__":
    unittest.main()
