from setuptools import setup, find_packages

setup(
    name="ML_Scratch",
    version="0.1",
    description="A library for simple machine learning model including: Linear Model (Regression + Classification), Non Linear Model (Regression + Classification), Binary Trees (Regression + Classification), SVM",
    packages=find_packages(),
    install_requires=["numpy"],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)