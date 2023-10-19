from setuptools import find_packages, setup

VERSION = '0.0.1'
DESCRIPTION = 'multi-fidelity gaussian process with radial basis function'
LONG_DESCRIPTION = 'A repo for multi-fidelity gaussian process with radial \
basis function surrogate as low-fidelity trend function '

# Setting up
setup(
    name="rbfgp",
    version=VERSION,
    author="Jiaxiang Yi (Delft University of Technology)",
    author_email="<J.Yi@tudelft.nl>",
    url='https://github.com/JiaxiangYi96/rbfgp.git',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    readme="README.md",
    install_requires='',
    keywords=['python', 'radial basis function', 'gaussian process'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
