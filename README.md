# genens
genens is an AutoML system for pipeline optimization based on developmental genetic programming.

## Installation
Clone the repository.
```
git clone https://github.com/gabrielasuchopar/genens.git
pip install genens
```


-----
## Using genens
As for now, the GenensClassifier is ready to be used. It has an interface similar to other scikit-learn estimators. When `fit()` is called, the evolutionary optimization is run. After it finishes, `predict()` produces a prediction with the best of optimized pipelines. Alternatively, you can call `get_best_pipelines()` to get pipelines from the pareto front.

```
from genens import GenensClassifier
from sklearn.datasets import load_iris()

iris = load_iris()
train_X, test_X, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.25)

clf = GenensClassifier()
clf.fit(train_X, train_y)
... # process of evolution

pred = clf.predict(test_X)
```

## Tests
Directory ./genens/tests contains scripts for running dataset tests and produce data about evolution process along with pickle files of best optimized pipelines. 
Sample config files are included in ``./genens/tests/config``.

- Run genEns on a dataset specified in the config file.

``python ./genens/tests/run_datasets.py --out OUT_DIR config CONFIG``

- Runs genEns on the [OpenML-CC18 benchmarking suite](https://www.openml.org/s/99)

``python ./genens/tests/run_openml.py --out OUT_DIR --config CONFIG``

More tests are to be included in later releases.
