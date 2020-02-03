import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="genens",
    version="0.1.1",
    author="Gabriela Suchoparova",
    author_email="gabi.suchoparova@gmail.com",
    description="A genetic AutoML system for ensemble methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabrielasuchopar/genens",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'deap',
        'graphviz',
        'joblib',
        'openml',
        'matplotlib',
        'numpy',
        'pygraphviz',
        'scikit-learn>=0.22',
        'seaborn',
        'stopit'
    ]
)
