import setuptools

setuptools.setup(
    name='ernie4us',
    version='0.88',
    description="Tensorflow access helper for Baidu's ERNIE NLP model",
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
