from setuptools import setup, find_packages

setup(
    name="dltools",
    version="0.0.6",
    description="Deep learning toolbox for training remotely.",
    author="Alexandre Boucaud",
    author_email="aboucaud@apc.in2p3.fr",
    packages=find_packages(),
    license="BSD",
    install_requires = [
        "numpy",
        "tensorflow",
        "h5py",
        "scikit-learn"
    ],
    python_requires='<=3.8',
)