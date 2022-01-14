# Installation instructions for developers #{pages-installation-dev}

## Standard dependencies

- Python 3
- GCC >=10 or Clang >=10
- CMake
- Make or Ninja

On Ubuntu or other Debian-based distributions:
```sh
sudo apt install python3 gcc-10 g++-10 cmake make
```
Make sure that a default version of GCC is selected:
```sh
sudo update-alternatives --remove-all gcc ||:
sudo update-alternatives --remove-all g++ ||:

sudo update-alternatives --install "/usr/bin/gcc" "gcc" "$(which gcc-10)" 100 \
                         --slave "/usr/bin/g++" "g++" "$(which g++-10)"
```
If you have other versions of GCC installed on your system, you can run `update-alternatives --install` once for each version, and then execute
```sh
sudo update-alternatives --config gcc
```
to select the correct version.

Alternatively, you could use the standard environment variables `CC`, `CXX`.

## Prepare virtual environment

From the root of the project directory tree:

```sh
python3 -m venv py-venv
. ./py-venv/bin/activate
```

## Install Python dependencies

```sh
pip install -r requirements.txt
```
## Install Eigen and Google Test

```sh
./scripts/install-eigen.sh      # https://eigen.tuxfamily.org/index.php
./scripts/install-gtest.sh      # https://google.github.io/googletest/
```

## Build tests

```sh
cmake -Bbuild -S. -DCMAKE_BUILD_TYPE=Asan
cmake --build build -j -t tests
./build/test/tests
```