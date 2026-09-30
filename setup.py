from setuptools import setup, find_packages

setup(
    packages=find_packages(where=".") + find_packages(where="Policy"),
    package_dir={
        "": ".",
        "hivebench": "Policy/hivebench",
        "deployment": "Policy/deployment",
        "Policy": "Policy",
    },
)