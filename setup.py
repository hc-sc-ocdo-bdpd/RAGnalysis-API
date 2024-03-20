from setuptools import setup, find_packages

setup(
    name='RAGnalysis-API',
    version='0.1.0',
    author='Health Canada CDO',
    description='A comprehensive library for our RAGnalysis API, including a=our Python client and Azure Function App code.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/hc-sc-ocdo-bdpd/RAGnalysis-API',
    packages=find_packages(),
    install_requires=[line.strip() for line in open('requirements.txt')],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
