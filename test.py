import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from utils import pp, visualize, to_json, show_all_variables
from models import ALOCC_Model
import matplotlib.pyplot as plt
from kh_tools import *
import numpy as np
import scipy.misc
from utils import *
import time
import os
from sklearn.metrics import roc_auc_score, average_precision_score
import math
import pickle
from configuration import Configuration as cfg

flags = tf.app.flags
flags.DEFINE_integer("nStride",1,"nStride ?[1]")
flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("attention_label", 1, "Conditioned label that growth attention of training label [1]")
flags.DEFINE_float("r_alpha", 0.2, "Refinement parameter [0.2]")
flags.DEFINE_integer("train_size", 5000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 45, "The size of image to use. [45]")
flags.DEFINE_integer("input_width", None, "The size of image to use. If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 45, "The size of the output images to produce [45]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "UCSD", "The name of dataset [UCSD, mnist]")
flags.DEFINE_string("dataset_address", "./dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test", "The path of dataset")
flags.DEFINE_string("input_fname_pattern", "*", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/UCSD_128_45_45/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def check_some_assertions():
    """
    to check some assertions in inputs and also check sth else.
    """
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    check_dir(FLAGS.checkpoint_dir)
    check_dir(FLAGS.sample_dir)

def main(_):
    print('Program is started at', time.clock())
#    pp.pprint(flags.FLAGS.__flags) # print all flags, suppress to unclutter output

    n_per_itr_print_results = 100
    n_fetch_data = 10
    kb_work_on_patch= False
    lst_test_dirs = ['Test004','Test005','Test006']
    
    n_stride = 10
    if FLAGS.dataset == 'UCSD':
        nd_input_frame_size = (240, 360)
        nd_patch_size = (45, 45)
        FLAGS.checkpoint_dir = "./checkpoint/UCSD_128_45_45/"
        FLAGS.dataset = 'UCSD'
        FLAGS.dataset_address = './dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'

    #DATASET PARAMETER : MNIST
    if FLAGS.dataset == 'mnist':
        FLAGS.dataset_address = './dataset/mnist'
        nd_input_frame_size = (28, 28)
        nd_patch_size = (28, 28)
        FLAGS.checkpoint_dir = "./checkpoint/mnist_128_28_28/"
        FLAGS.input_width = nd_patch_size[0]
        FLAGS.input_height = nd_patch_size[1]
        FLAGS.output_width = nd_patch_size[0]
        FLAGS.output_height = nd_patch_size[1]

    if FLAGS.dataset == 'bdd100k':
        nd_input_frame_size = (FLAGS.input_height, FLAGS.input_width)
        nd_patch_size = nd_input_frame_size
        FLAGS.checkpoint_dir = "checkpoint/{}_{}_{}_{}".format(
        FLAGS.dataset, FLAGS.batch_size,
        FLAGS.output_height, FLAGS.output_width)
    
    log_dir = "./log/"+cfg.dataset+"/"+cfg.architecture + "/"
    FLAGS.sample_dir = log_dir

    check_some_assertions()

    nd_patch_size = (FLAGS.input_width, FLAGS.input_height)
    FLAGS.nStride = n_stride

    #FLAGS.input_fname_pattern = '*'
    FLAGS.train = False
    FLAGS.epoch = 1


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    run_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config.gpu_options.allow_growth=True
    with tf.Session(config=run_config) as sess:
        tmp_ALOCC_model = ALOCC_Model(
                    sess,
                    input_width=FLAGS.input_width,
                    input_height=FLAGS.input_height,
                    output_width=FLAGS.output_width,
                    output_height=FLAGS.output_height,
                    batch_size=FLAGS.batch_size,
                    sample_num=FLAGS.batch_size,
                    attention_label=FLAGS.attention_label,
                    r_alpha=FLAGS.r_alpha,
                    is_training=FLAGS.train,
                    dataset_name=FLAGS.dataset,
                    dataset_address=FLAGS.dataset_address,
                    input_fname_pattern=FLAGS.input_fname_pattern,
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    sample_dir=FLAGS.sample_dir,
                    nd_patch_size=nd_patch_size,
                    n_stride=n_stride,
                    n_per_itr_print_results=n_per_itr_print_results,
                    kb_work_on_patch=kb_work_on_patch,
                    nd_input_frame_size = nd_input_frame_size,
                    n_fetch_data=n_fetch_data)

#        show_all_variables()


        print('--------------------------------------------------')
        print('Loading pretrained model from ',tmp_ALOCC_model.checkpoint_dir,'...')
        tmp_ALOCC_model.f_check_checkpoint()

        if FLAGS.dataset=='mnist':
            #mnist = input_data.read_data_sets(FLAGS.dataset_address)
            mnist = tf.keras.datasets.mnist
            (x_train, y_train),(x_test, y_test) = mnist.load_data() 

            inlier_idx = tmp_ALOCC_model.attention_label
            specific_idx = np.where(y_test == inlier_idx)[0]
            inlier_data = x_test[specific_idx].reshape(-1, 28, 28, 1)


            anomaly_frac = 0.5
            potential_idx_anomaly = np.where(y_test != inlier_idx)[0]
            specific_idx_anomaly = [potential_idx_anomaly[x] for x in
                                   random.sample(range(0, len(potential_idx_anomaly)), math.ceil(anomaly_frac*len(specific_idx)/(1-anomaly_frac)))]

            anomaly_data = x_test[specific_idx_anomaly].reshape(-1, 28, 28, 1)
            data = np.append(inlier_data, anomaly_data).reshape(-1, 28, 28, 1)

            # True labels are 1 for inliers and 0 for anomalies, since discriminator outputs higher values for inliers
            labels = np.append(np.ones(len(inlier_data)),np.zeros(len(anomaly_data)))

            # Shuffle data so not only anomaly points are removed if data is shortened below
            tmp_perm = np.random.permutation(len(data))
            data = data[tmp_perm]
            labels = labels[tmp_perm]

            # Only whole batches
            n_batches = len(data)//tmp_ALOCC_model.batch_size
#            print("Batch size: ", tmp_ALOCC_model.batch_size, "n batches: ", n_batches)
            data = data[:n_batches*tmp_ALOCC_model.batch_size]
            labels = labels[:len(data)]

            # Get test results from discriminator
            results_d = tmp_ALOCC_model.f_test_frozen_model(data)

            # Compute performance metrics
            roc_auc = roc_auc_score(labels, results_d)
            print('AUROC: ',roc_auc)

            roc_prc = average_precision_score(labels, results_d)
            print("AUPRC: ", roc_prc)

            print('Test completed')
            exit()
            #generated_data = tmp_ALOCC_model.feed2generator(data[0:FLAGS.batch_size])
        elif FLAGS.dataset == 'UCSD':
            # else in UCDS (depends on infrustructure)
            for s_image_dirs in sorted(glob(os.path.join(FLAGS.dataset_address, 'Test[0-9][0-9][0-9]'))):
                tmp_lst_image_paths = []
                if os.path.basename(s_image_dirs) not in ['Test004']:
                    print('Skip ',os.path.basename(s_image_dirs))
                    continue
                for s_image_dir_files in sorted(glob(os.path.join(s_image_dirs + '/*'))):
                    if os.path.basename(s_image_dir_files) not in ['068.tif']:
                        print('Skip ', os.path.basename(s_image_dir_files))
                        continue
                    tmp_lst_image_paths.append(s_image_dir_files)


                #random
                #lst_image_paths = [tmp_lst_image_paths[x] for x in random.sample(range(0, len(tmp_lst_image_paths)), n_fetch_data)]
                lst_image_paths = tmp_lst_image_paths
                #images =read_lst_images(lst_image_paths,nd_patch_size,nd_patch_step,b_work_on_patch=False)
                images = read_lst_images_w_noise2(lst_image_paths, nd_patch_size, nd_patch_step)

                lst_prob = process_frame(os.path.basename(s_image_dirs),images,tmp_ALOCC_model)

                print('pseudocode test is finished')

                # This code for just check output for readers
                # ...
        
        elif FLAGS.dataset in ('prosivic','dreyeve', 'bdd100k'):
            data = tmp_ALOCC_model.data
            labels = tmp_ALOCC_model.test_labels

        # Below is done for all datasets
        test_dir = log_dir + "test/"
        check_dir(test_dir)

        # Shuffle data so not only anomaly points are removed if data is shortened below
        tmp_perm = np.random.permutation(len(data))
        data = data[tmp_perm]
        labels = labels[tmp_perm]

        # Only whole batches
        n_batches = len(data)//tmp_ALOCC_model.batch_size
#            print("Batch size: ", tmp_ALOCC_model.batch_size, "n batches: ", n_batches)
        data = data[:n_batches*tmp_ALOCC_model.batch_size]
        labels = labels[:len(data)]

        # Get test results from discriminator
        results_d = tmp_ALOCC_model.f_test_frozen_model(data)

        # Compute performance metrics
        roc_auc = roc_auc_score(labels, results_d)
        print('AUROC: ',roc_auc)

        roc_prc = average_precision_score(labels, results_d)
        print("AUPRC: ", roc_prc)

        # Pickle results
        results = [labels, results_d]
        results_file = test_dir + "results.pkl"
        with open(results_file,'wb') as f:
            pickle.dump(results,f)
            
        print('Test completed')
        exit()
        #generated_data = tmp_ALOCC_model.feed2generator(data[0:FLAGS.batch_size])

def process_frame(s_name,frames_src,sess):
    nd_patch,nd_location = get_image_patches(frames_src,sess.patch_size,sess.patch_step)
    frame_patches = nd_patch.transpose([1,0,2,3])
    print('frame patches :{}\npatches size:{}'.format(len(frame_patches),(frame_patches.shape[1],frame_patches.shape[2])))

    lst_prob = sess.f_test_frozen_model(frame_patches)

    #  This code for just check output for readers
    # ...

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    tf.app.run()


