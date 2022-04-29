import setuptools

# to build and load:
# 1. python setup.py bdist_wheel
# 2. python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/pylineaGT-X.X.X-py3-none-any.whl
# 3. python -m twine upload dist/pylineaGT-X.X.X-py3-none-any.whl

setuptools.setup(
    name = "pylineaGT",
    version = "0.0.8",
    author = "Elena Buscaroli",
    description = "",
    package_dir = {"":"src"},
    packages = setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires = [
        "pandas==1.3.3",
        "pyro-api==0.1.2",
        "pyro-ppl==1.8.0",
        "torch==1.11.0",
        "numpy==1.20.3",
        "scikit-learn==1.0.2",
    ],
)
