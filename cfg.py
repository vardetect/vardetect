
import tensorflow as tf
# tf.disable_v2_behavior()

flags = tf.app.flags

# Parameters
# ==================================================

# Data loading params

# Model Hyperparameters
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("lm_dropout_keep_prob", 0.9, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_string("optimizer", "adam", "Custom optimizer")

# TextCNN
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 64)")
tf.flags.DEFINE_integer("sequence_length", 300, "seq length")
tf.flags.DEFINE_integer("vocab_size", 10000, "vocab size")
tf.flags.DEFINE_string("copy_filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("copy_num_filters", 100, "Number of filters per filter size (default: 64)")

# RNN/LSTM
tf.flags.DEFINE_integer("mem_size", 128, "mem size")
tf.flags.DEFINE_float("lm_learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_integer("lm_early_stop_tolerance", 10, "Early stop (default: 200 evaluations)")


#Text
tf.flags.DEFINE_string("padding_loc", "right", "padding loc")
tf.flags.DEFINE_integer("embedding_size", 300, "emb size)")
tf.flags.DEFINE_boolean("use_pretrained_embeddings", False, "uss pretrined embeddings)")
tf.flags.DEFINE_boolean("trainable_embeddings", False, "is embeddings trainable")
tf.flags.DEFINE_boolean("lm_trainable_embeddings", False, "is embeddings trainable")



# Secret model Training parameters
tf.flags.DEFINE_boolean("train_source_model", False, "Train the source model")
tf.flags.DEFINE_boolean("use_recon_input", False, "use reconstructed input")
tf.flags.DEFINE_string("defender_type", None,  "defender type")

tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 150)")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many steps (default: 1)")
tf.flags.DEFINE_integer("gan_evaluate_every", 100, "GAN evaluate every")
tf.flags.DEFINE_integer("early_stop_tolerance", 10, "Early stop (default: 20 evaluations)")

tf.flags.DEFINE_integer("copy_early_stop_tolerance", 10, "Early stop (default: 20 evaluations)")
tf.flags.DEFINE_integer("dae_early_stop_tolerance", 50, "Early stop (default: 10 evaluations)")

tf.flags.DEFINE_boolean("transfer_attack_activethief", False, "transferability attacks using activethief")
tf.flags.DEFINE_boolean("transfer_attack_jbda", False, "transferability attacks using papernot")
tf.flags.DEFINE_boolean("transfer_attack_tramer", False, "transferability attacks using papernot")


#Substitute model Training params
tf.flags.DEFINE_integer("copy_num_epochs", 1000, "Number of training epochs (default: 1000)")
tf.flags.DEFINE_integer("copy_evaluate_every", 1, "Evaluate copy model after this many epochs")
tf.flags.DEFINE_boolean("copy_one_hot", True, "Copy using one hot")

tf.flags.DEFINE_boolean("extract_model_activethief", False, "extract secret model using activethief")
tf.flags.DEFINE_boolean("extract_model_jbda", False, "extract secret model using papernot")
tf.flags.DEFINE_boolean("extract_model_tramer", False, "extract secret model using papernot")


tf.flags.DEFINE_boolean("train_svm", False, "train svm")
tf.flags.DEFINE_boolean("train_defender", False, "defend_model")
tf.flags.DEFINE_boolean("ignore_invalid_class", True, "ignore invalid class")

tf.flags.DEFINE_float("svm_threshold", None, "svm threshold")

tf.flags.DEFINE_float("anomaly_ratio", 1.0, "anomaly ratio")

tf.flags.DEFINE_boolean("filtered_attack", False, "do not use ood detector")

tf.flags.DEFINE_float("eps", 0.1, "eps")
tf.flags.DEFINE_string("jtype", 'jsma',  "jsma")

# VAE params
tf.flags.DEFINE_float("noise_mean", 5.0, "noise mean")


# GAN Params
tf.flags.DEFINE_float("disc_threshold", 0.1, "Disc threshold")
tf.flags.DEFINE_float("recon_threshold", 0.1, "Recon threshold")
tf.flags.DEFINE_integer("gan_epochs", 1000, "GAN epochs")

# DAE Test param
tf.flags.DEFINE_float("vnoise", 0.2, "noise volume")

#SVM params
tf.flags.DEFINE_float("vnoise_min", 0.2, "noise volume min")
tf.flags.DEFINE_float("vnoise_max", 1.0, "noise volume max")
tf.flags.DEFINE_float("C", 2.0, "C")
tf.flags.DEFINE_float("gamma", 0.001, "C")




#VAE params
tf.flags.DEFINE_integer("latent_dim", 8, "latent dim")


tf.flags.DEFINE_integer("grad0_prate", 10, "grad0_prate")


# GPU Parameters
tf.flags.DEFINE_boolean("allow_gpu_growth", True, "Allow gpu growth")
tf.flags.DEFINE_boolean("allow_soft_placement", False, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")


tf.flags.DEFINE_string("source_model", None,  "Source model type(eg DeepCNN/DNN) to copy")
tf.flags.DEFINE_string("copy_model", None,  "Copy model type(eg DeepCNN/DNN)")
tf.flags.DEFINE_string("true_dataset", None,  "Source model will be trained on this")
tf.flags.DEFINE_string("noise_dataset", None, "Source model will be copied using this")
tf.flags.DEFINE_string("sampling_method", "random", "sampling method")

tf.flags.DEFINE_integer("kmeans_iter", 100, "K means iteration")

tf.flags.DEFINE_integer("phase1_fac", 5, "Multiple of samples to use in Phase 1")
tf.flags.DEFINE_integer("phase2_fac", 10, "Multiple of samples to use in Phase 2")


tf.flags.DEFINE_integer("phase1_size", 20000, "Multiple of samples to use in Phase 1")
tf.flags.DEFINE_integer("phase2_size", 10000, "Multiple of samples to use in Phase 2")


tf.flags.DEFINE_integer("subsampling_start_batch", 1, "Start Batch of imagenet to use for subsampling experiments")
tf.flags.DEFINE_integer("subsampling_end_batch", 1, "End Batch of imagenet to use for subsampling experiments")

tf.flags.DEFINE_integer("num_to_keep", 100000, "Number of samples to make use of for imagenet")

tf.flags.DEFINE_integer("query_budget", None, "total query budget")
tf.flags.DEFINE_integer("initial_seed", None, "intial seed")
tf.flags.DEFINE_integer("num_iter", None, "num of iterations")
tf.flags.DEFINE_integer("val_size", 1000, "validation size")
tf.flags.DEFINE_integer("k", None, "add queries")

tf.flags.DEFINE_integer("seed", 1337, "seed for RNGs")


tf.flags.DEFINE_integer("linesearch_budget", 10000, "validation size")


# Hack for dealing with Jupyter Notebook
tf.flags.DEFINE_string("f", "f", "f")


from os.path import expanduser
tf.flags.DEFINE_string("home", expanduser("~"), "Home directory")


cfg = tf.app.flags.FLAGS



config                          = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = cfg.allow_gpu_growth
config.log_device_placement     = cfg.log_device_placement   
config.allow_soft_placement     = cfg.allow_soft_placement   
