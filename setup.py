from distutils.core import setup

setup(
    name='Reduction-Dask',
    version='0.1.0',
    author='Joshua Chung',
    packages=['reduction-dask'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description="A package which utilizes Dask distributed package for testing and tuning dimensionality reduction techniques.",
    install_requires= [
        'hdbscan',
        'numpy',
        'pandas',
        'shap',
        'lhsmdu',
        'dask[complete]',
        'distributed',
        'scikit-learn-extra',
        'scikit-learn',
        'scipy',
        'ace'
    ]
)