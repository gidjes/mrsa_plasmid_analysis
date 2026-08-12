from setuptools import setup, find_packages

setup(
    name="mge_bootstrap",
    version="1.0",
    packages=find_packages(),  # finds scripts/
    install_requires=[
        "numpy",
        "pandas",
        "seaborn",
        "scikit-learn"
    ],
    entry_points={"console_scripts": ["mge_bootstrap = scripts.main:main"]},
)
