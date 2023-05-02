from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='tfilterpy',
    version='0.0.3',
    project_urls={
        'Logo': 'https://raw.githubusercontent.com/LeparaLaMapara/tfilterpy/main/branding/logo/tfilters-logo.jpeg'
    },
    author='Thabang L. Mashinini- Sekgoto, Lebogang M. Mashinini-Sekgoto, Palesa D. Mashinini-Sekgoto',
    author_email='thabangline@gmail.com',
    description='This package is for Bayesian filtering models.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # long_description=open('README.md').read(),
    # long_description='This package is for Bayesian filtering models.',
    url='https://github.com/leparalamapara/tfilterpy',
    packages=find_packages(),
    install_requires=[
        'numpy',
        # other dependencies
    ],
    python_requires='>=3.6',
    # Example section
    examples=[
        'examples/motion-estimation-kalmanfilters.ipynb'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)