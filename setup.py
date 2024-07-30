# setup.py
from setuptools import setup, find_packages

setup(
    name='ProHabana',
    version='0.1',
    install_requires=[
        # List your dependencies here
    ],
    author='Aditya Kulshrestha',
    author_email='kulshresthaaditya02@gmail.com',
    description='Easy to profile Habana Gaudi workload package',
    long_description=open('README.md').read(),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    long_description_content_type='text/markdown',
    url='',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
