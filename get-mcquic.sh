#!/bin/sh
curl -fsSL ? >> /tmp/mcquic.yml
conda create -f /tmp/mcquic.yml
pip install mcquic
