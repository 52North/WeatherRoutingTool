#!/bin/bash

if [ -z "$1" ]
then
  ENV_FILE='.env'
else
  ENV_FILE=$1
fi

if [ ! -f ${ENV_FILE} ]; then
    echo "File ${ENV_FILE} not found!"
    return
fi

echo "Setting environment variables from ${ENV_FILE} file"
export $(grep -v '^#' ${ENV_FILE} | xargs)