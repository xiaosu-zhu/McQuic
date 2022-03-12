C:\Miniconda3\condabin\conda config --set anaconda_upload yes

C:\Miniconda3\condabin\conda-build -c conda-forge -c bioconda -c pytorch -c xiaosu-zhu --output-folder . .
