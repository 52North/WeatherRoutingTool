#!/bin/bash

pwd=${PWD}

cd ${WRT_PATH}
echo 'moving to path ' ${PWD}
export $(grep -v '^#' .env | xargs)
cd ${pwd}
echo 'back in path ' ${PWD}
