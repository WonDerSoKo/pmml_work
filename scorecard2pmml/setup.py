from setuptools import setup, find_packages


setup(
    name = "scorecard2pmml",
    version = "1.0.2",
    author = 'Simon Reon',
    author_email = '18946198329@163.com',
    packages = ['scorecard2pmml'],
    license = 'GPL-2',
    description = 'transform bin & preprocessing table of scorecard model into pmml file',
    install_requires=['numpy','pandas','tabulate','lxml'],
    #目标文件
    py_modules = 'scorecard2pmml.py')