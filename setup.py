from setuptools import setup, find_packages

setup(
    name="RUC_Generator",          # Package name used for imports
    version="0.1.0",
    packages=find_packages(),      # Finds all packages recursively
    install_requires=[             # Dependencies (can match requirements.txt)
        "numpy",
        "opencv-python-headless",
        "pandas",
        "pillow",
        "plotly",
        "streamlit",
        "scipy",
        "scikit-image",
        "scikit-learn",
    ],
    python_requires='>=3.9',       # Match your Python version
)