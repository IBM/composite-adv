import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="composite_adv",
    version="0.0.1",
    author="Lei Hsiung",
    author_email="hsiung@m109.nthu.edu.tw",
    description="Composite Adversarial Attack",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/twweeb/composite-adv",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'kornia',
        'numpy',
        'requests',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)