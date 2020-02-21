import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ernie4us',
    version='0.1.0',
    description="Tensorflow accessible helper for Baidu's ERNIE NLP model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['tensorflow>=1.13.1', 'dataclasses'],
    url='https://github.com/hotpads/ERNIE-for-the-rest-of-us/',
    author='Winston Quock',
    license='Apache License 2.0',
    packages=setuptools.find_packages(),
    zip_safe=True,
    classifiers=[
         "Programming Language :: Python :: 3"
    ]
)
