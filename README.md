This is a preliminary code repository for the paper "Stateful Detection of Model Extraction Attacks".

## Setup

We bundle the Fashion-MNIST dataset and trained models for the Fashion-MNIST dataset.

To run the code, set up a virtual environment:

    $ virtualenv env --python python2

Once the environment is set up, source it:

    $ source ./env/bin/activate
    
Then, install the required packages: 

    $ python2 -m pip install -r requirements.txt
    
If you are not using a virtual environment, it may be a good idea to use the --user flag during this step.

Following this, run:

    $ jupyter notebook

This will cause a browser window to open with the demo notebooks.

To execute the notebooks, open them, and then use the Jupyter menu to run them (in the numbered order):

    Execute > Restart and Run All.