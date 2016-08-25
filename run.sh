#!/bin/bash

virtualenv py
/code/py/bin/pip install -r requirements.txt

/code/py/bin/python main.py
