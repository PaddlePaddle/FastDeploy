CURRENT_DIR=${PWD}

cd cpp
doxygen

cd ../python
make html

cd ${CURRENT_DIR}
