from setuptools import setup, find_packages

setup(
    name='tfilterpy',
    version='0.0.2',
    project_urls={
        'Logo': 'https://raw.githubusercontent.com/LeparaLaMapara/tfilterpy/main/branding/logo/tfilters-logo.jpeg'
    },
    author='Thabang L. Mashinini- Sekgoto, Lebogang M. Mashinini-Sekgoto, Palesa D. Mashinini-Sekgoto',
    author_email='thabangline@gmail.com',
    description='This package is for Bayesian filtering models.',
    long_description=open('README.md').read(),
    # long_description='This package is for Bayesian filtering models.',
    url='https://github.com/leparalamapara/tfilterpy',
    packages=find_packages(),
    install_requires=[
        'numpy',
        # other dependencies
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)