#!/bin/bash

pwd=${PWD}
run_modus=$1

if [ "${run_modus}" == 'test' ]
then
  WRT_PATH=${pwd}'/WeatherRoutingTool'
else
  if ! [ -n "${WRT_PATH}" ]
  then
    echo 'Please set the path to the base folder of the WeatherRoutingTool package'
  fi
fi

echo 'WRT_PATH is set to ' ${WRT_PATH}

cd ${WRT_PATH}

if [ "${run_modus}" == 'test' ]
then
  echo 'Setting environment variables for test case'
  export $(grep -v '^#' .env.test | xargs)
else
  echo 'Setting environment variables for normal use case'
  export $(grep -v '^#' .env | xargs)
fi

cd ${pwd}


