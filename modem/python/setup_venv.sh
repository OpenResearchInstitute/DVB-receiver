#!/bin/bash

echo "Starting setup of Python virtual enviroment..."

# create the virtual enviroment
echo "Installing the system packages, please enter your password"
sudo apt-get install python3-virtualenv
sudo apt-get install python3-numpy

# start the virtual enviroment and install the required packages
echo "Starting virtual enviroment..."
python3 -m venv venv && source venv/bin/activate && pip install wheel && pip install -r requirements.txt

# symbolically link the CocoTB drivers
cd venv/lib/python3.*/site-packages/cocotb/drivers
mv amba.py amba_default.py
ln -s ../../../../../../cocotb/amba.py
echo "CocoTB library files have been symbolically linked."