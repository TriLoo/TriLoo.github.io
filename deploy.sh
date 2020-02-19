#!/bin/sh

set -e

yarn docs:build

cd docs/.vuepress/dist

git init
git add -A
git commit -m 'deploy'

git push -f git@github.com:TriLoo/TriLoo.github.io.git master

cd -

