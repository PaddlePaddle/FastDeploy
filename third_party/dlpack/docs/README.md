# Documentation for DLPack

## Building Locally

The following dependencies must be downloaded in order to build docs:

- Doxygen (On debian distros, simply run `sudo apt -y install doxygen`)
- Python dependencies present in the `doc_requirements.txt` file in the root directory of the project. Run `python3 -m pip install -r doc_requirements.txt` to install them.

Once the dependencies are installed, docs can be built using either CMake or the Makefile from the root directory of the project.

- Using Makefile: Run `make doc` to build the HTML pages. Run `make show_docs` to serve the website locally.
- Using CMake: Build with `BUILD_DOCS` option `ON`: `mkdir -p build && cd build && cmake .. -DBUILD_DOCS=ON && make`
