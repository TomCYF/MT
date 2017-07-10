#!/bin/bash
export PYTHONIOENCODING=utf-8
echo "Running mt experiment"
source activate mtenv
python nmt_translate.py
echo "Finished training mt model"

