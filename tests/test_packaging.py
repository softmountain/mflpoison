import unittest

from setuptools import find_namespace_packages


class PackagingTest(unittest.TestCase):
    def test_source_distribution_discovers_both_runtime_packages(self):
        packages = set(find_namespace_packages(exclude=["tests*"]))
        self.assertIn("mflpoison", packages)
        self.assertIn("fed_multimodal", packages)
        self.assertIn("fed_multimodal.dataloader", packages)


if __name__ == "__main__":
    unittest.main()
