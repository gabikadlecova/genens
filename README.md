# Genens
Genens is an AutoML system for pipeline optimization based on developmental genetic programming.

## Installation
Clone the repository.
```
git clone https://github.com/gabrielasuchopar/genens.git
cd genens
```

Set up a conda environment:

on Windows
```
conda env create -f environment-win.yml
```

on Linux
```
conda env create -f environment-linux.yml
```

Finally, add ``path-to-repository/genens/`` to PYTHONPATH.

-----
There may be some problems regarding the pygraphviz package on Windows. You can contact me if any problems occur.

## Using Genens
As for now, the GenensClassifier is fully functional. You can use it just as any scikit-learn classifier. When fit is called,
it finds performs evolutionary optimization.

```
from genens import GenensClassifier
from sklearn.datasets import load_iris()

iris = load_iris()
train_X, test_X, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.25)

clf = GenensClassifier()
clf.fit(train_X, train_y)
... # process of evolution

score = clf.predict(test_X)
```

## Tests
You can run tests which produce data about evolution and pickle files of best optimized pipelines. 
Sample config files are included in ``./genens/tests/config``.

- Run Genens on a dataset specified in the config file.

``python ./genens/tests/run_datasets.py --out OUT_DIR config CONFIG``

- Runs Genens on the [OpenML-CC18 benchmarking suite](https://www.openml.org/s/99)

``python ./genens/tests/run_openml.py --out OUT_DIR --config CONFIG``
