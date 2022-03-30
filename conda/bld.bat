$NPY_NUM_BUILD_JOBS = 8
"%PYTHON%" setup.py install
if errorlevel 1 exit 1
