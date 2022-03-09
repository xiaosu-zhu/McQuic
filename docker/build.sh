#!/bin/sh

tmpdir=$(mktemp -d)

cp ../environment.yml $tmpdir
cp Dockerfile $tmpdir && cd $tmpdir

docker build -t xiaosuzhu/mcquic:latest .
