import unittest

from fastdeploy import ModelRegistry
from fastdeploy.plugins import load_model_register_plugins


class TestModelRegistryPlugins(unittest.TestCase):
    def test_plugin_registers_one_architecture(self):
        """Test that loading plugins registers exactly one new architecture."""
        initial_archs = set(ModelRegistry.get_supported_archs())
        print("Supported architectures before loading plugins:", sorted(initial_archs))

        # Load plugins
        load_model_register_plugins()

        final_archs = set(ModelRegistry.get_supported_archs())
        print("Supported architectures after loading plugins:", sorted(final_archs))

        added_archs = final_archs - initial_archs
        added_count = len(added_archs)

        # verify
        self.assertEqual(
            added_count,
            1,
            f"Expected exactly 1 new architecture to be registered by plugins, "
            f"but {added_count} were added: {sorted(added_archs)}",
        )


if __name__ == "__main__":
    unittest.main()
