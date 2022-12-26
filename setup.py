import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='MiSeCom',
    packages=['MiSeCom'],
    version='0.1.1',
    description='Detect if the English has missing sentence components such as Subject, Verb, Object',
    author='Le Minh Khoi',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    py_modules=["MiSeCom"],
    install_requires=['torch', 'pytorch-lightning', 'wandb', 'transformers', 'datasets', 'tokenizers']
)