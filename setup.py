from setuptools import setup, find_packages
from os.path import join, dirname, splitext, basename
from glob import glob


def read(name):
    return open(join(dirname(__file__), name)).read()


setup(
    name='pypsr',
    version='0.0.1',
    license='MIT',

    description='Phase-space reconstruction in Python.',
    long_description=read('README.rst'),

    author='Henry S. Harrison',
    author_email='henry.schafer.harrison@gmail.com',

    url='https://bitbucket.org/hharrison/pypsr',
    download_url='https://bitbucket.org/hharrison/pypsr/get/default.tar.gz',

    package_dir={'': 'src'},
    packages=find_packages('src'),
    py_modules=[splitext(basename(i))[0] for i in glob(join('src', '*.py'))],

    keywords='phase-space reconstruction psr Taken Takens embedding ami nonlinear dynamics chaos chaotic attractor',

    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Utilities',
        'Topic :: Scientific/Engineering',
    ],

    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'toolz',
    ],
)
