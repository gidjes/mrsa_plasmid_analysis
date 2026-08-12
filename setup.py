from setuptools import setup, find_packages

setup(
    name="mrsa_plasmid_analysis",
    version="1.0",
    packages=find_packages(),  # finds scripts/
    install_requires=[
        "numpy",
        "pandas",
        "seaborn",
        "scikit-learn"
        "pip"
    ],
    entry_points={"console_scripts": ["mrsa_plasmid_analysis = scripts.main:main"]},
)
