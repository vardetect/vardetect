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
    
## Training Code

To train the MLaaS and substitute models yourself, we provide a console script `run.py`.  
You can use this file in the following ways:

1. Train the MLaaS (victim) model:

        $ python run.py --true_dataset fashionmnist --train_source_model
    
2. Train the VAE:

        $ python run.py --true_dataset fashionmnist --defender_type vae --train_defender
    
3. Perform synthetic extaction, with or without the defense in place:

        $ python run.py --true_dataset fashionmnist --extract_model_tramer --query_budget 10000
        $ python run.py --true_dataset fashionmnist --defender_type vae --extract_model_tramer --query_budget 10000
    
4. Perform AdvPD extaction, with or without the defense in place, e.g.:
   
   **Papernot**
   
        $ python run.py --true_dataset fashionmnist --extract_model_jbda --jtype jsma --eps 0.1 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --defender_type vae --extract_model_jbda --jtype jsma --eps 0.1 --query_budget 10000
    
   **PRADA (N-FGSM)** 
   
        $ python run.py --true_dataset fashionmnist --extract_model_jbda --jtype n-fgsm --eps 0.25 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --defender_type vae --extract_model_jbda --jtype n-fgsm --eps 0.25 --query_budget 10000
    
   **PRADA (N-IFGSM)** 
    
        $ python run.py --true_dataset fashionmnist --extract_model_jbda --jtype n-ifgsm --eps 0.25 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --defender_type vae --extract_model_jbda --jtype n-ifgsm --eps 0.25 --query_budget 10000
    
   **PRADA (T-RND-FGSM)** 
    
        $ python run.py --true_dataset fashionmnist --extract_model_jbda --jtype t-rnd-fgsm --eps 0.25 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --defender_type vae --extract_model_jbda --jtype t-rnd-fgsm --eps 0.25 --query_budget 10000
    
   **PRADA (T-RND-IFGSM)** 
    
        $ python run.py --true_dataset fashionmnist --extract_model_jbda --jtype t-rnd-ifgsm --eps 0.25 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --defender_type vae --extract_model_jbda --jtype t-rnd-ifgsm --eps 0.25 --query_budget 10000
    
   **PRADA (Color)** 
    
        $ python run.py --true_dataset fashionmnist --extract_model_jbda --jtype color --eps 0.1 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --defender_type vae --extract_model_jbda --jtype color --eps 0.1 --query_budget 10000
    
5. Perform NPD extraction, with or without the defense in place, e.g.:

        $ python run.py --true_dataset fashionmnist --extract_model_activethief --sampling_method random --num_iter 0 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --extract_model_activethief --sampling_method uncertainty --num_iter 0 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --extract_model_activethief --sampling_method kcenter --num_iter 10 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --extract_model_activethief --sampling_method adversarial --num_iter 10 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --extract_model_activethief --sampling_method adversarial-kcenter --num_iter 10 --query_budget 10000
    
        $ python run.py --true_dataset fashionmnist --defender_type vae --extract_model_activethief --sampling_method random --num_iter 0 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --defender_type vae --extract_model_activethief --sampling_method uncertainty --num_iter 10 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --defender_type vae --extract_model_activethief --sampling_method kcenter --num_iter 10 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --defender_type vae --extract_model_activethief --sampling_method adversarial --num_iter 10 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --defender_type vae --extract_model_activethief --sampling_method adversarial-kcenter --num_iter 10 --query_budget 10000
    
6. Check Transferability for all of the above cases, e.g.

        $ python run.py --true_dataset fashionmnist --transfer_attack_tramer --query_budget 10000
        $ python run.py --true_dataset fashionmnist --transfer_attack_jbda --jtype jsma --eps 0.1 --query_budget 10000
        $ python run.py --true_dataset fashionmnist --transfer_attack_activethief --sampling_method random --num_iter 0 --query_budget 10000

7. The attacker's budget can be varied by changing the value passed to `run.py`.

## Dataset

In addition to the Fashion-MNIST dataset (fashionmnist), the following other datasets are supported: mnist, gtsr, streetview.  
You must first download the following datasets and place them in your home directory (`/home/<your-username>/datasets/streetview`, etc.):

[Google Drive link](https://drive.google.com/drive/folders/12tmkkDPx5ZjwGyDpv2R7vqFTbxu_a6HE?usp=sharing)

Note that the full imagenet dataset is also required for the NPD extraction attacks (change the path in `dsl/imagenet_dsl.py` to point to your home directory after extraction).