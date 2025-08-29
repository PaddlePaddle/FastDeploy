import unittest

import fastdeploy


class TestVersion(unittest.TestCase):
    def test_get_version(self):
        ver = fastdeploy.version()
        assert ver.count("COMMIT") > 0


if __name__ == "__main__":
    unittest.main()
