#!/bin/bash

compress() {
  path=$1
  pkg_name=${path//"/"/"_"}
  echo "compressing $pkg_name"
  rm -rf output/$pkg_name
  mkdir -p output/$pkg_name/example
  cp -r $1/* output/$pkg_name/example
  cp *$2.md output/$pkg_name
  cd output
  tar czvf $pkg_name.tgz $pkg_name/
  cd -
}

for first in *
do
  if [ -d "$first" ] && [ "$first" != "output" ]
  then
    echo "found $first"
    for second in $first/*
    do
      if [ -d "$second" ]
      then
        dir_name=${second##*/}
        if [ "$dir_name" == "serving" ]
        then
          compress $second "serving"
        else
          for third in $second/*
          do
            compress $third ${third##*/}
          done
        fi
      fi
    done
  fi
done
