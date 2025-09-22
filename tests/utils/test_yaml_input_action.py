import argparse
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from fastdeploy.utils import YamlInputAction


class TestYamlInputAction(unittest.TestCase):
    def setUp(self):
        self.parser = MagicMock(spec=argparse.ArgumentParser)
        self.namespace = argparse.Namespace()
        self.action = YamlInputAction(option_strings=[], dest="config")

    def test_call_with_yaml_file(self):
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = """
            key1: value1
            key2: value2
            """
            f.write(yaml_content)
            f.close()

            # Test
            self.action(self.parser, self.namespace, f.name)

            # Verify
            self.assertEqual(self.namespace.config, {"key1": "value1", "key2": "value2"})

            # Clean up
            os.unlink(f.name)

    def test_call_with_yaml_string(self):
        yaml_str = """
        key1: value1
        key2: value2
        """

        self.action(self.parser, self.namespace, yaml_str)
        self.assertEqual(self.namespace.config, {"key1": "value1", "key2": "value2"})

    def test_call_with_invalid_yaml(self):
        with self.assertRaises(ValueError):
            self.action(self.parser, self.namespace, "invalid")

    def test_call_with_non_dict_yaml(self):
        with self.assertRaises(ValueError):
            self.action(self.parser, self.namespace, "- item1\n- item2")

    def test_call_with_existing_config(self):
        # Set existing config
        self.namespace.config = {"existing": "value"}

        yaml_str = """
        new_key: new_value
        """

        self.action(self.parser, self.namespace, yaml_str)
        self.assertEqual(self.namespace.config, {"existing": "value", "new_key": "new_value"})


if __name__ == "__main__":
    unittest.main()
