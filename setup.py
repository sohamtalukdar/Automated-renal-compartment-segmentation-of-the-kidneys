from setuptools import setup

with open("requirements.txt", "r") as file:
    dependencies = file.readlines()

setup(
    name='Renal Segmentation using Convolutional Neural Network',
    version='0.1',
    author = "Soham Talukdar",
    install_requires=dependencies
)
