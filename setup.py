import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="genens",
    version="0.1.16",
    author="Gabriela Suchoparova",
    author_email="gabi.suchoparova@gmail.com",
    description="A genetic AutoML system for ensemble methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabrielasuchopar/genens",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        '': ['.logging_config.json', '*.yaml']
    },
    python_requires='>=3.6',
    install_requires=[
        'deap',
        'joblib',
        'openml',
        'matplotlib',
        'numpy',
        'scikit-learn>=0.22',
        'seaborn',
        'stopit',
        'pyyaml'
    ]
)
