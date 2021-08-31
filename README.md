# reduction-dask

In order to setup the proper environment to run package, please install these depedencies:

module load python
conda create --name dask_environment --clone lazy-mpi4py
source activate numeraidask0
conda install dask distributed ipykernel numpy pandas scikit-learn scipy matplotlib ten
pip install hdbscan
pip install pynndescent
pip install lhsmdu
pip install umap-learn 
pip install umap
pip install scikit-learn-extra
pip install shap
pip install tbb
conda install -c conda-forge --no-deps dask-mpi

