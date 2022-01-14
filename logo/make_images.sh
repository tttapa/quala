#!/usr/bin/env bash

cd "$( dirname "${BASH_SOURCE[0]}" )"

inkscape quala_favicon.pdf --pdf-poppler --export-text-to-path --export-plain-svg -o quala_favicon.svg
inkscape quala_logo.pdf --pdf-poppler --export-text-to-path --export-plain-svg -o quala_logo.svg
inkscape quala_favicon.pdf --pdf-poppler --export-type png -w 256 -h 256
