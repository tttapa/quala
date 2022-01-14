#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"/../..

set -ex

./scripts/install-eigen.sh Release
./scripts/install-gtest.sh
