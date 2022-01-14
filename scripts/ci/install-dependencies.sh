#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"/../..

set -ex

./scripts/install-eigen.sh
./scripts/install-gtest.sh
