from setuptools import setup, find_packages

setup(
    name="FlexAutoML",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "catboost",
        "lightgbm",
        "scikit-learn",
    ],
    author='Hengzhe Zhang',
    author_email='hengzhe.zhang@ecs.vuw.ac.nz',
    description='A machine learning toolbox',
)
