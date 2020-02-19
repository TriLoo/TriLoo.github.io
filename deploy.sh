#!/bin/sh

set -e

yarn docs:build

cd docs/.vuepress/dist

git init
git add -A
git commit -m 'deploy'

git remote add origin git@github.com:TriLoo/TriLoo.github.io.git
git push -f origin master

cd -

