#!/bin/bash

if [ -z "$1" ]
then
      echo "Usage: ./inference.sh \"your message here\""
else
      python scripts/inference.py "$1"
fi
