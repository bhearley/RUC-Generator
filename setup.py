from setuptools import setup, find_packages

setup(
    name="RUCGenerator",          # Name you will use in imports
    version="0.1.0",
    packages=find_packages(),        # Automatically finds all packages
    install_requires=[
        "opencv-python-headless",
        "numpy",
        "pandas",
        "pillow",
        "plotly",
        "streamlit",
        "scipy",
        "scikit-image",
        "scikit-learn",
    ],             
)