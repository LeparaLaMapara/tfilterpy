from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tfilterpy",
    version="1.0.0",
    author="Thabang L. Mashinini-Sekgoto",
    author_email="thabangline@gmail.com",
    description="A Python package for Bayesian filtering models such as Kalman and Particle Filters.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://leparalamapara.github.io/tfilterpy/",
    project_urls={
        "Documentation": "https://leparalamapara.github.io/tfilterpy/",
        "Source": "https://github.com/LeparaLaMapara/tfilterpy",
        "Tracker": "https://github.com/LeparaLaMapara/tfilterpy/issues",
        "Logo": "https://raw.githubusercontent.com/LeparaLaMapara/tfilterpy/main/branding/logo/tfilters-logo.jpeg",
    },
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "dask>=2023.5.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        "": ["examples/*.ipynb"],  # Include Jupyter notebooks in the package
    },
    include_package_data=True,
)
