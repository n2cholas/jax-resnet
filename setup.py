import io
import os
import re

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get('encoding', 'utf8')) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r'^__version__ = ["\']([^"\']*)["\']', version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


readme = read('README.md')

VERSION = find_version('jax_resnet', '__init__.py')

requirements = ['jax', 'jaxlib', 'flax']

setup(
    # Metadata
    name='jax-resnet',
    version=VERSION,
    author='Nicholas Vadivelu',
    author_email='nicholas.vadivelu@gmail.com',
    url='https://github.com/n2cholas/jax-resnet',
    description='Framework-agnostic library for checking array shapes at runtime.',
    long_description_content_type='text/markdown',
    long_description=readme,
    license='MIT',
    # Package info
    packages=find_packages(exclude=(
        'tests',
        'tests.*',
    )),
    zip_safe=True,
    install_requires=requirements,
)
