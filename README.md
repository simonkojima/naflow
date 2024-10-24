# naflow
Neurophysiological Data Workflow

# test
```
python -m unittest discover -s tests
python -m unittest discover -s tests -p "test_*.py"
```

# sphinx
```
sphinx-apidoc -M -E -f -o ./docs ./naflow
sphinx-build ./docs ./docs/_build -a

sphinx-apidoc -M -E -f -o ./docs ./naflow && sphinx-build ./docs ./docs/_build -a
```