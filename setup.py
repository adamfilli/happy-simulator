from setuptools import setup, find_packages

setup(
    name="happy-simulator",
    version="0.1.0",
    description="Discrete event simulation tool (Matlab SimEvent for Python)",
    author="adamfilli",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "pandas",
        "pygetwindow",
        "screeninfo"
    ],
    include_package_data=True,
    python_requires=">=3.13",
)
