from __future__ import division
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import re
from ops import *
from utils import *
from kh_tools import *
import logging
import matplotlib.pyplot as plt
from loadbdd100k import load_bdd100k_data_attribute_spec, load_bdd100k_data_filename_list
from configuration import Configuration as cfg
from keras.preprocessing.image import load_img, img_to_array

class ALOCC_Model(object):
  def __init__(self, sess,
               input_height=28,input_width=28, output_height=28, output_width=28,
               batch_size=128, sample_num = 32, attention_label=1, is_training=True,
               z_dim=100, gf_dim=64, df_dim=64, gfc_dim=512, dfc_dim=512, c_dim=3,
               dataset_name=None, dataset_address=None, input_fname_pattern=None,
               checkpoint_dir=None, log_dir=None, sample_dir=None, r_alpha = 0.2,
               kb_work_on_patch=True, nd_input_frame_size=(240, 360), nd_patch_size=(10, 10), n_stride=1,
               n_fetch_data=10, n_per_itr_print_results=500):
    """
    This is the main class of our Adversarially Learned One-Class Classifier for Novelty Detection
    :param sess: TensorFlow session      
    :param batch_size: The size of batch. Should be specified before training. [128]
    :param attention_label: Conditioned label that growth attention of training label [1]
    :param r_alpha: Refinement parameter [0.2]        
    :param z_dim:  (optional) Dimension of dim for Z. [100] 
    :param gf_dim: (optional) Dimension of gen filters in first conv layer. [64] 
    :param df_dim: (optional) Dimension of discrim filters in first conv layer. [64] 
    :param gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024] 
    :param dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024] 
    :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]  
    :param sample_dir: Directory address which save some samples [.]
    :param kb_work_on_patch: Boolean value for working on PatchBased System or not [True]
    :param nd_input_frame_size:  Input frame size 
    :param nd_patch_size:  Input patch size
    :param n_stride: PatchBased data preprocessing stride
    :param n_fetch_data: Fetch size of Data 
    :param n_per_itr_print_results: # of printed iteration   
    """

    self.n_per_itr_print_results=n_per_itr_print_results
    self.nd_input_frame_size = nd_input_frame_size
    self.b_work_on_patch = kb_work_on_patch
    self.sample_dir = sample_dir

    self.sess = sess
    self.is_training = is_training

    self.r_alpha = r_alpha

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')
    self.d_bn4 = batch_norm(name='d_bn4')
    self.d_bn5 = batch_norm(name='d_bn5')
    self.d_bn6 = batch_norm(name='d_bn6')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')
    self.g_bn4 = batch_norm(name='g_bn4')
    self.g_bn5 = batch_norm(name='g_bn5')
    self.g_bn6 = batch_norm(name='g_bn6')
    self.g_bn7 = batch_norm(name='g_bn7')
    self.g_bn8 = batch_norm(name='g_bn8')
    self.g_bn9 = batch_norm(name='g_bn9')
    self.g_bn10 = batch_norm(name='g_bn10')
    self.g_bn11 = batch_norm(name='g_bn11')

    self.g_bn = batch_norm(name = 'general_generator_bn')

    self.dataset_name = dataset_name
    self.dataset_address= dataset_address
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.log_dir = log_dir

    self.attention_label = attention_label

    if self.is_training:
      logging.basicConfig(filename='ALOCC_loss.log', level=logging.INFO)

    if self.dataset_name == 'mnist':
      mnist = tf.keras.datasets.mnist
      #mnist = input_data.read_data_sets(self.dataset_address)
      (x_train, y_train),(x_test, y_test) = mnist.load_data()
      specific_idx = np.where(y_train == self.attention_label)[0]
      print("Inlier digit: ", self.attention_label)
      self.data = x_train[specific_idx].reshape(-1, 28, 28, 1)
      self.c_dim = 1
      print("Training data length: ", len(self.data))

    elif self.dataset_name == 'UCSD':
      self.nStride = n_stride
      self.patch_size = nd_patch_size
      self.patch_step = (n_stride, n_stride)
      lst_image_paths = []
      for s_image_dir_path in glob(os.path.join(self.dataset_address, self.input_fname_pattern)):
        for sImageDirFiles in glob(os.path.join(s_image_dir_path+'/*')):
          lst_image_paths.append(sImageDirFiles)
      self.dataAddress = lst_image_paths
      lst_forced_fetch_data = [self.dataAddress[x] for x in random.sample(range(0, len(lst_image_paths)), n_fetch_data)]

      self.data = lst_forced_fetch_data
      self.c_dim = 1

    elif self.dataset_name == "bdd100k":
      self.c_dim = 3
      self.dataAddress = cfg.img_folder
      if self.is_training:
        self.data, _, _, _ = load_bdd100k_data_filename_list(cfg.img_folder, cfg.norm_filenames, cfg.out_filenames, cfg.n_train, cfg.n_val, cfg.n_test, cfg.out_frac, self.input_height, self.input_width, self.c_dim, get_norm_and_out_sets=False, shuffle=cfg.shuffle)
      else: # load testdata
        _ , _, self.data, self.test_labels = load_bdd100k_data_filename_list(cfg.img_folder, cfg.norm_filenames, cfg.out_filenames, cfg.n_train, cfg.n_val, cfg.n_test, cfg.out_frac, self.input_height, self.input_width, self.c_dim, get_norm_and_out_sets=False, shuffle=cfg.shuffle)
        self.test_labels = 1 - self.test_labels
    elif self.dataset_name in ("dreyeve", "prosivic"):
      # Load data
      self.c_dim = 3
      self.dataAdress = None

      if self.is_training:
        self.data = [img_to_array(load_img(Cfg.train_folder + filename)) for filename in os.listdir(Cfg.train_folder)][:n_train]
        # self._X_val = [img_to_array(load_img(Cfg.prosivic_val_folder + filename)) for filename in os.listdir(Cfg.prosivic_val_folder)][:Cfg.prosivic_n_val] 
      else: #load test data     
      n_test_out = Cfg.prosivic_n_test - Cfg.prosivic_n_test_in
      _X_test_in = [img_to_array(load_img(Cfg.prosivic_test_in_folder + filename)) for filename in os.listdir(Cfg.prosivic_test_in_folder)][:Cfg.prosivic_n_test_in]
      _X_test_out = [img_to_array(load_img(Cfg.prosivic_test_out_folder + filename)) for filename in os.listdir(Cfg.prosivic_test_out_folder)][:n_test_out]
      _y_test_in  = np.ones((Cfg.prosivic_n_test_in,),dtype=np.int32)
      _y_test_out = np.zeros((n_test_out,),dtype=np.int32)
      self.data = np.concatenate([_X_test_in, _X_test_out])
      self.test_labels = np.concatenate([_y_test_in, _y_test_out])
      # self.out_frac = Cfg.out_frac

    else:
      assert('Error in loading dataset')

    self.grayscale = (self.c_dim == 1)
    self.build_model()

  # =========================================================================================================
  def build_model(self):
    image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    inputs = self.inputs
    sample_inputs = self.sample_inputs

    self.z = tf.placeholder(tf.float32,[self.batch_size] + image_dims, name='z')

    self.G = self.generator(self.z)
    self.D, self.D_logits = self.discriminator(inputs)

    self.sampler = self.sampler(self.z)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    # tesorboard setting
    # self.z_sum = histogram_summary("z", self.z)
    #self.d_sum = histogram_summary("d", self.D)
    #self.d__sum = histogram_summary("d_", self.D_)
    #self.G_sum = image_summary("G", self.G)

    # Simple GAN's losses
    self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

    # Refinement loss
    self.g_r_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G,labels=self.z))
    self.g_loss  = self.g_loss + self.g_r_loss * self.r_alpha
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]


# =========================================================================================================
  def train(self, config):
    d_optim = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.g_loss, var_list=self.g_vars)

    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()


    self.saver = tf.train.Saver()

    self.g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])

    log_dir = os.path.join(self.log_dir, self.model_dir)
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)

    self.writer = SummaryWriter(log_dir, self.sess.graph)

    if config.dataset == 'mnist':
      sample = self.data[0:self.sample_num]
    elif config.dataset =='UCSD':
      if self.b_work_on_patch:
        sample_files = self.data[0:10]
      else:
        sample_files = self.data[0:self.sample_num]
      sample,_ = read_lst_images(sample_files, self.patch_size, self.patch_step, self.b_work_on_patch)
      sample = np.array(sample).reshape(-1, self.patch_size[0], self.patch_size[1], 1)
      sample = sample[0:self.sample_num]
    elif config.dataset == 'bdd100k':
      sample = self.data[0:self.sample_num]

    # export images
    sample_inputs = np.array(sample).astype(np.float32)
    scipy.misc.imsave('./{}/train_input_samples.jpg'.format(config.sample_dir), montage(sample_inputs[:,:,:,0]))

    # load previous checkpoint
    counter = 1
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")


    # load traning data
    if config.dataset == 'mnist':
      sample_w_noise = get_noisy_data(self.data)
    if config.dataset == 'UCSD':
      sample_files = self.data
      sample, _ = read_lst_images(sample_files, self.patch_size, self.patch_step, self.b_work_on_patch)
      sample = np.array(sample).reshape(-1, self.patch_size[0], self.patch_size[1], 1)
      sample_w_noise,_ = read_lst_images_w_noise(sample_files, self.patch_size, self.patch_step)
      sample_w_noise = np.array(sample_w_noise).reshape(-1, self.patch_size[0], self.patch_size[1], 1)
    if config.dataset in ('bdd100k','dreyeve','prosivic'):
      sample_w_noise = get_noisy_data(self.data)

    for epoch in xrange(config.epoch):
      print('Epoch ({}/{})-------------------------------------------------'.format(epoch+1,config.epoch))
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size
      elif config.dataset == 'UCSD':
        batch_idxs = min(len(sample), config.train_size) // config.batch_size
      elif config.dataset == 'bdd100k':
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      # for detecting valuable epoch that we must stop training step
      # sample_input_for_test_each_train_step.npy
#      sample_test = np.load('SIFTETS.npy').reshape([self.batch_size,self.input_height,self.input_width,self.c_dim])[0:self.batch_size]

      for idx in xrange(0, batch_idxs):
        if config.dataset == 'mnist':
          batch = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
          batch_noise = sample_w_noise[idx * config.batch_size:(idx + 1) * config.batch_size]
        elif config.dataset == 'UCSD':
          batch = sample[idx * config.batch_size:(idx + 1) * config.batch_size]
          batch_noise = sample_w_noise[idx * config.batch_size:(idx + 1) * config.batch_size]
        elif config.dataset == 'bdd100k':
          batch = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
          batch_noise = sample_w_noise[idx * config.batch_size:(idx + 1) * config.batch_size]

        batch_images = np.array(batch).astype(np.float32)
        batch_noise_images = np.array(batch_noise).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

        if config.dataset == 'mnist':
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
                                         feed_dict={ self.inputs: batch_images, self.z: batch_noise_images })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
                                         feed_dict={ self.z: batch_noise_images })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
                                         feed_dict={ self.z: batch_noise_images })
          self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({ self.z: batch_noise_images })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          errG = self.g_loss.eval({self.z: batch_noise_images})
        else:
          # update discriminator
          _, summary_str = self.sess.run([d_optim, self.d_sum],
                                          feed_dict={ self.inputs: batch_images, self.z: batch_noise_images })
          self.writer.add_summary(summary_str, counter)

          # update refinement(generator)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
                                          feed_dict={ self.z: batch_noise_images })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
                                          feed_dict={ self.z: batch_noise_images })
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({ self.z: batch_noise_images })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          errG = self.g_loss.eval({self.z: batch_noise_images})

        counter += 1

        msg = "Epoch:[%2d][%4d/%4d]--> d_loss: %.8f, g_loss: %.8f" % (epoch+1, idx+1, batch_idxs, errD_fake+errD_real, errG)
        print(msg)
        logging.info(msg)

        '''
        if np.mod(counter, self.n_per_itr_print_results) == 0:
          if config.dataset == 'mnist':
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_inputs,
                  self.inputs: sample_inputs
              }
            )
            manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
            manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
            save_images(samples, [manifold_h, manifold_w],
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
          # ====================================================================================================
          else:
            #try:
              samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_inputs,
                    self.inputs: sample_inputs,
                },
              )

              sample_test_out = self.sess.run(
                [self.sampler],
                feed_dict={
                    self.z: sample_test
                },
              )
              # export images
              scipy.misc.imsave('./{}/z_test_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx),
                            montage(samples[:, :, :, 0]))

              # export images
              scipy.misc.imsave('./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx),
                                montage(samples[:, :, :, 0]))

              msg = "[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)
              print(msg)
              logging.info(msg)
              '''

      self.save(config.checkpoint_dir, epoch)

  # =========================================================================================================
  def discriminator(self, image,reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      if self.dataset_name in ["mnist", "UCSD"]:
      # df_dim defaults to 64
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
        h5 = tf.nn.sigmoid(h4,name='d_output')
        
        return h5, h4

      elif self.dataset_name == "bdd100k": # for 256x256 images
        h0 = lrelu(conv2d(image, self.df_dim/4         , name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim/2 , name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim  ,  name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim* 2, name='d_h3_conv')))
        h4 = lrelu(self.d_bn4(conv2d(h3, self.df_dim* 4, name='d_h4_conv')))
        h5 = lrelu(self.d_bn5(conv2d(h4, self.df_dim* 8, name='d_h5_conv')))
        h6 = lrelu(self.d_bn6(conv2d(h5, self.df_dim*16, name='d_h6_conv')))
        h7 = linear(tf.reshape(h6, [self.batch_size, -1]), 1, 'd_h6_lin')
        h8 = tf.nn.sigmoid(h7,name='d_output')

        return h8, h7
      
      elif self.dataset_name in ("prosivic", "dreyeve"):
        tmp = cfg.architecture.split("_")
        use_pool = int(tmp[0]) == 1 # 1 or 0
        n_conv = int(tmp[1])
        n_dense = int(tmp[2])
        c_in = int(tmp[3])
        zsize = int(tmp[4])
        ksize= int(tmp[5])
        stride = int(tmp[6])
        pad = int(tmp[7])
        num_filters = c_in

        if use_pool:
          # Compute output padding and padding so that output size is same as input for deconv-layers
          if stride != 1:
            print("Warning: stride not 1 while using pooling. Algorithm is not built to support this")
          else:
            outpad = (ksize-stride)%2
          pad = (ksize-stride+outpad)//2
        else: # using stride to increase image size: set pad and outpad so that h_out = stride * h_in
          if stride == 1:
            print("No pool and stride = 1. Image size will not increase")
          else:
            outpad = (ksize-stride)%2
            pad = (ksize-stride+outpad)//2

        # pad = (pad,pad) # needs to be specified for both dimensions

        x = image
        # Do first n_conv-1 conv layers (last one might be different)
        for i in range(n_conv-1):
          x = conv2d(x, num_filters, k_h=ksize, k_w=ksize, d_h=stride, d_w=stride, pad = pad, name="d_conv2d_"+str(i+1))
          print("x: ",x.shape)
          if cfg.use_batchnorm:
            bn = batch_norm(name = 'g_bn_'+str(i+1))
            x = bn(x)
          if use_pool:
            x = pool(x)
          x = lrelu(x)
          num_filters *= 2

        if n_dense > 0:
          # Add one more convlayer as above
          x = conv2d(x, num_filters, k_h=ksize, k_w=ksize, d_h=stride, d_w=stride, pad = pad, name="d_conv2d_"+str(n_conv-1))
          print("x: ",x.shape)
          if cfg.use_batchnorm:
            bn = batch_norm(name="g_bn_"+str(n_conv-1))
            x = bn(x)
          if use_pool:
            x = pool(x)
          x = lrelu(x)
          num_filters *= 2 # TODO: Remove this?

          print("into dense: ", x.shape)
          x = linear(tf.reshape(h6, [self.batch_size, -1]), 1, 'd_lin')

        else:
          # Final conv-layer is different
            h = cfg.image_height // (2**(n_conv-1))
            x = conv2d(x, num_filters, k_h=h, k_w=h, d_h=1, d_w=1, name= 'd_conv2d_encode', pad = 0)
          
        return tf.nn.sigmoid(x,name='d_output')



  # =========================================================================================================
  def generator(self, z):
    with tf.variable_scope("generator") as scope:
      
      if self.dataset_name in ['mnist', 'UCSD']:

        # Compute output shapes for decoding steps
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # Encode
        print("Input shape: ", z.shape)
        hae0 = lrelu(self.g_bn4(conv2d(z   , self.df_dim * 2, name='g_encoder_h0_conv')))
        print("x: ", hae0.shape)
        hae1 = lrelu(self.g_bn5(conv2d(hae0, self.df_dim * 4, name='g_encoder_h1_conv')))
        print("x: ", hae1.shape)
        hae2 = lrelu(self.g_bn6(conv2d(hae1, self.df_dim * 8, name='g_encoder_h2_conv')))
        print("x: ", hae2.shape)

        # Decode
        h2, self.h2_w, self.h2_b = deconv2d(
          hae2, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_decoder_h1', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))
        print("x: ", h2.shape)
        h3, self.h3_w, self.h3_b = deconv2d(
          h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_decoder_h0', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))
        print("x: ", h3.shape)
        h4, self.h4_w, self.h4_b = deconv2d(
          h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_decoder_h00', with_w=True)
        print("x: ", h4.shape)
        return tf.nn.tanh(h4,name='g_output')

      elif self.dataset_name == 'bdd100k':

        # Compute output shapes for decoding steps
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
        s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)

        # Encode
        hae0 = lrelu(self.g_bn6(conv2d(z   , self.df_dim // 4, name='g_encoder_h0_conv')))
        hae1 = lrelu(self.g_bn7(conv2d(hae0, self.df_dim // 2, name='g_encoder_h1_conv')))
        hae2 = lrelu(self.g_bn8(conv2d(hae1, self.df_dim    , name='g_encoder_h2_conv')))
        hae3 = lrelu(self.g_bn9(conv2d(hae2, self.df_dim * 2, name='g_encoder_h3_conv')))
        hae4 = lrelu(self.g_bn10(conv2d(hae3, self.df_dim * 4, name='g_encoder_h4_conv')))
        hae5 = lrelu(self.g_bn11(conv2d(hae4, self.df_dim * 8, name='g_encoder_h5_conv')))

        # Decode
        h1, self.h1_w, self.h1_b = deconv2d(
          hae5, [self.batch_size, s_h32, s_w32, self.gf_dim*8], name='g_decoder_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2, self.h2_w, self.h2_b = deconv2d(
          h1, [self.batch_size, s_h16, s_w16, self.gf_dim*4], name='g_decoder_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
          h2,   [self.batch_size, s_h8, s_w8,   self.gf_dim*2], name='g_decoder_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
          h3,   [self.batch_size, s_h4, s_w4,   self.gf_dim   ], name='g_decoder_h4', with_w=True)
        h4 = tf.nn.relu(self.g_bn4(h4))
        #print("s_h2:",s_h2," s_w2:", s_w2, "type: ", type(s_h2))
        #print("dim: ", self.gf_dim/2, "type: ", type(self.gf_dim/2))
        h5, self.h5_w, self.h5_b = deconv2d(
          h4,   [self.batch_size, s_h2, s_w2,   self.gf_dim//2], name='g_decoder_h5', with_w=True)
        h5 = tf.nn.relu(self.g_bn5(h5))

        h6, self.h6_w, self.h6_b = deconv2d(
          h5,   [self.batch_size, s_h, s_w,     self.c_dim],    name='g_decoder_h6', with_w=True)

        return tf.nn.tanh(h6,name='g_output')

      elif self.dataset_name in ("prosivic", "dreyeve"):
        tmp = cfg.architecture.split("_")
        use_pool = int(tmp[0]) == 1 # 1 or 0
        n_conv = int(tmp[1])
        n_dense = int(tmp[2])
        c_out = int(tmp[3])
        zsize = int(tmp[4])
        ksize= int(tmp[5])
        stride = int(tmp[6])
        pad = int(tmp[7])
        num_filters = c_out

        if use_pool:
          # Compute output padding and padding so that output size is same as input for deconv-layers
          if stride != 1:
            print("Warning: stride not 1 while using pooling. Algorithm is not built to support this")
          else:
            outpad = (ksize-stride)%2
          pad = (ksize-stride+outpad)//2
        else: # using stride to increase image size: set pad and outpad so that h_out = stride * h_in
          if stride == 1:
            print("No pool and stride = 1. Image size will not increase")
          else:
            outpad = (ksize-stride)%2
            pad = (ksize-stride+outpad)//2

        # pad = (pad,pad) # needs to be specified for both dimensions

        # Encode
        x = z
        # Do first n_conv-1 conv layers (last one might be different)
        for i in range(n_conv-1):
          x = conv2d(x, num_filters, k_h=ksize, k_w=ksize, d_h=stride, d_w=stride, pad = pad, name="g_conv2d_"+str(i+1))
          print("x: ",x.shape)
          if cfg.use_batchnorm:
            bn = batch_norm(name = 'g_bn_'+str(i+1))
            x = bn(x)
          if use_pool:
            x = pool(x)
          x = lrelu(x)
          num_filters *= 2

        if n_dense > 0:
          # Add one more convlayer as above
          x = conv2d(x, num_filters, k_h=ksize, k_w=ksize, d_h=stride, d_w=stride, pad = pad, name="g_conv2d_"+str(n_conv-1))
          print("x: ",x.shape)
          if cfg.use_batchnorm:
            bn = batch_norm(name="g_bn_"+str(n_conv-1))
            x = bn(x)
          if use_pool:
            x = pool(x)
          x = lrelu(x)
          num_filters *= 2 # TODO: Remove this?

          linear(tf.reshape(h6, [self.batch_size, -1]), zsize, 'g_lin_encode')

        else:
          # Final conv-layer is different
            h = cfg.image_height // (2**(n_conv-1))
            encoded = conv2d(x, num_filters, k_h=h, k_w=h, d_h=1, d_w=1, name="g_conv2d_encode", pad = 0)
          
        print("Encoded: ", encoded.shape)
      
        # Decode

        if n_dense > 0:
          h1 = self.image_height // (2**n_conv) # height = width of image going into first conv layer
          num_filters =  c_out * (2**(n_conv-1))
          n_dense_out = h1**2 * num_filters
          x = linear(tf.reshape(encoded,[self.batch_size,-1]),num_filters, name = 'g_lin_decode')
          # x = x.permute(0,3,1,2)

          x = tf.reshape(x,[self.batch_size,h1,h1,num_filters]
          x = F.relu(x)
          num_filters //= 2

          if use_pool:
            x = upscale(x)
          # First deconv-layer (identical to rest, add here just to get right number of layers)
          h = h1*2
          output_shape = [self.batch_size, h, h, num_filters]
          x = deconv2d(x,output_shape, k_h = ksize, k_w = ksize, d_h = stride, d_w = stride,name = 'g_deconv2d_1')
        
        else: # No dense layer, upscale to correct shape with unique first deconv-layer
          x = encoded
          h2 = self.image_height // (2**(n_conv-1)) # height of image going in to second deconv layer
          num_filters = c_out * (2**(n_conv-2))
          h = h2
          output_shape = [self.batch_size, h, h, num_filters]
          x = deconv2d(x, output_shape,  k_h = h, k_w = h, d_h = 1, d_w = 1, name = 'g_deconv2d_1')
          bn = batch_norm(name = 'g_bn_d_1')
          x = lrelu(bn(x))
          
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] //=2

        for i in range(n_conv - 2): # Add all but last conv layer
          if use_pool:
            x = upscale(x)

          bn = batch_norm(name = 'g_bn_d_'+str(i+2))  
          x = lrelu(bn(deconv2d(x, output_shape,  k_h = ksize, k_w = ksize, d_h = stride, d_w = stride, name = 'g_deconv2d_'+str(i+2))))

          output_shape[1] *= 2
          output_shape[2] *= 2
          output_shape[3] //=2

        if use_pool:
          x = upscale(x)

        x = deconv2d(x, output_shape,  k_h = ksize, k_w = ksize, d_h = stride, d_w = stride, name = 'g_deconv2d_reconstruct')

        x = tf.nn.tanh(self.output_layer(x), name = 'g_output')

        return x

  # =========================================================================================================
  def sampler(self, z, y=None): # identical code to "generator" above
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if self.dataset_name in ['mnist', 'UCSD']:

        # Compute output shapes for decoding steps
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # Encode
        hae0 = lrelu(self.g_bn4(conv2d(z   , self.df_dim * 2, name='g_encoder_h0_conv')))
        hae1 = lrelu(self.g_bn5(conv2d(hae0, self.df_dim * 4, name='g_encoder_h1_conv')))
        hae2 = lrelu(self.g_bn6(conv2d(hae1, self.df_dim * 8, name='g_encoder_h2_conv')))

        # Decode
        h2, self.h2_w, self.h2_b = deconv2d(
          hae2, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_decoder_h1', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
          h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_decoder_h0', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
          h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_decoder_h00', with_w=True)

        return tf.nn.tanh(h4,name='g_output')

      elif self.dataset_name == 'bdd100k':

        # Compute output shapes for decoding steps
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
        s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)

        # Encode
        hae0 = lrelu(self.g_bn6(conv2d(z   , self.df_dim // 4, name='g_encoder_h0_conv')))
        hae1 = lrelu(self.g_bn7(conv2d(hae0, self.df_dim // 2, name='g_encoder_h1_conv')))
        hae2 = lrelu(self.g_bn8(conv2d(hae1, self.df_dim    , name='g_encoder_h2_conv')))
        hae3 = lrelu(self.g_bn9(conv2d(hae2, self.df_dim * 2, name='g_encoder_h3_conv')))
        hae4 = lrelu(self.g_bn10(conv2d(hae3, self.df_dim * 4, name='g_encoder_h4_conv')))
        hae5 = lrelu(self.g_bn11(conv2d(hae4, self.df_dim * 8, name='g_encoder_h5_conv')))

        # Decode
        h1, self.h1_w, self.h1_b = deconv2d(
          hae5, [self.batch_size, s_h32, s_w32, self.gf_dim*8], name='g_decoder_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2, self.h2_w, self.h2_b = deconv2d(
          h1, [self.batch_size, s_h16, s_w16, self.gf_dim*4], name='g_decoder_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
          h2,   [self.batch_size, s_h8, s_w8,   self.gf_dim*2], name='g_decoder_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
          h3,   [self.batch_size, s_h4, s_w4,   self.gf_dim   ], name='g_decoder_h4', with_w=True)
        h4 = tf.nn.relu(self.g_bn4(h4))

        h5, self.h5_w, self.h5_b = deconv2d(
          h4,   [self.batch_size, s_h2, s_w2,   self.gf_dim//2], name='g_decoder_h5', with_w=True)
        h5 = tf.nn.relu(self.g_bn5(h5))

        h6, self.h6_w, self.h6_b = deconv2d(
          h5,   [self.batch_size, s_h, s_w,     self.c_dim],    name='g_decoder_h6', with_w=True)

        return tf.nn.tanh(h6,name='g_output')

      elif self.dataset_name in ("prosivic", "dreyeve"):
        tmp = cfg.architecture.split("_")
        use_pool = int(tmp[0]) == 1 # 1 or 0
        n_conv = int(tmp[1])
        n_dense = int(tmp[2])
        c_out = int(tmp[3])
        zsize = int(tmp[4])
        ksize= int(tmp[5])
        stride = int(tmp[6])
        pad = int(tmp[7])
        num_filters = c_out

        if use_pool:
          # Compute output padding and padding so that output size is same as input for deconv-layers
          if stride != 1:
            print("Warning: stride not 1 while using pooling. Algorithm is not built to support this")
          else:
            outpad = (ksize-stride)%2
          pad = (ksize-stride+outpad)//2
        else: # using stride to increase image size: set pad and outpad so that h_out = stride * h_in
          if stride == 1:
            print("No pool and stride = 1. Image size will not increase")
          else:
            outpad = (ksize-stride)%2
            pad = (ksize-stride+outpad)//2

        # pad = (pad,pad) # needs to be specified for both dimensions

        # Encode
        x = z
        # Do first n_conv-1 conv layers (last one might be different)
        for i in range(n_conv-1):
          x = conv2d(x, num_filters, k_h=ksize, k_w=ksize, d_h=stride, d_w=stride, pad = pad, name="g_conv2d_"+str(i+1))
          print("x: ",x.shape)
          if cfg.use_batchnorm:
            bn = batch_norm(name = 'g_bn_'+str(i+1))
            x = bn(x)
          if use_pool:
            x = pool(x)
          x = lrelu(x)
          num_filters *= 2

        if n_dense > 0:
          # Add one more convlayer as above
          x = conv2d(x, num_filters, k_h=ksize, k_w=ksize, d_h=stride, d_w=stride, pad = pad, name="g_conv2d_"+str(n_conv-1))
          print("x: ",x.shape)
          if cfg.use_batchnorm:
            bn = batch_norm(name="g_bn_"+str(n_conv-1))
            x = bn(x)
          if use_pool:
            x = pool(x)
          x = lrelu(x)
          num_filters *= 2 # TODO: Remove this?

          x = linear(tf.reshape(x,[self.batch_size,-1]), zsize, name = 'g_lin_encode')
        else:
          # Final conv-layer is different
            h = cfg.image_height // (2**(n_conv-1))
            encoded = conv2d(x, num_filters, k_h=h, k_w=h, d_h=1, d_w=1, name="g_conv2d_encode", pad = 0)
          
        print("Encoded: ", encoded.shape)
      
        # Decode

        if n_dense > 0:
          h1 = self.image_height // (2**n_conv) # height = width of image going into first conv layer
          num_filters =  c_out * (2**(n_conv-1))
          n_dense_out = h1**2 * num_filters
          x = linear(tf.reshape(encoded,[self.batch_size,-1]),num_filters, name = 'g_lin_decode')

          x = tf.reshape(x,[self.batch_size,h1,h1,num_filters])
          x = F.relu(x)
          num_filters //= 2

          if use_pool:
            x = upscale(x)
          # First deconv-layer (identical to rest, add here just to get right number of layers)
          h = h1*2
          output_shape = [self.batch_size, h, h, num_filters]
          x = deconv2d(x,output_shape, k_h = ksize, k_w = ksize, d_h = stride, d_w = stride,name = 'g_deconv2d_1')
        
        else: # No dense layer, upscale to correct shape with unique first deconv-layer
          x = encoded
          h2 = self.image_height // (2**(n_conv-1)) # height of image going in to second deconv layer
          num_filters = c_out * (2**(n_conv-2))
          h = h2
          output_shape = [self.batch_size, h, h, num_filters]
          x = deconv2d(x, output_shape,  k_h = h, k_w = h, d_h = 1, d_w = 1, name = 'g_deconv2d_1')
          bn = batch_norm(name = 'g_bn_d_1')
          x = lrelu(bn(x))
          
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] //=2

        for i in range(n_conv - 2): # Add all but last conv layer
          if use_pool:
            x = upscale(x)

          bn = batch_norm(name = 'g_bn_d_'+str(i+2))  
          x = lrelu(bn(deconv2d(x, output_shape,  k_h = ksize, k_w = ksize, d_h = stride, d_w = stride, name = 'g_deconv2d_'+str(i+2))))

          output_shape[1] *= 2
          output_shape[2] *= 2
          output_shape[3] //=2

        if use_pool:
          x = upscale(x)

        x = deconv2d(x, output_shape,  k_h = ksize, k_w = ksize, d_h = stride, d_w = stride, name = 'g_deconv2d_reconstruct')

        x = tf.nn.tanh(self.output_layer(x), name = 'g_output')
        
        return x
  # =========================================================================================================
  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  # =========================================================================================================
  def save(self, checkpoint_dir, step):
    model_name = "ALOCC_Model.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  # =========================================================================================================
  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

  # =========================================================================================================

  def f_check_checkpoint(self):
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()
    print(" [*] Reading checkpoints...")
    self.saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      could_load = True
      checkpoint_counter = counter
    else:
      print(" [*] Failed to find a checkpoint at specified location: ", self.checkpoint_dir)
      could_load = False
      checkpoint_counter =0

    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
      return counter
    else:
      print(" [!] Load failed...")
      return -1

  # =========================================================================================================
  def f_test_frozen_model(self,lst_image_slices=[]):
    lst_generated_img= []
    lst_discriminator_v = []
    tmp_shape = lst_image_slices.shape
    if self.dataset_name=='UCSD':
      tmp_lst_slices = lst_image_slices.reshape(-1, tmp_shape[2], tmp_shape[3], 1)
    else:
      tmp_lst_slices = lst_image_slices
    batch_idxs = len(tmp_lst_slices) // self.batch_size

    print('start new process ...')
    for i in xrange(0, batch_idxs):
        batch_data = tmp_lst_slices[i * self.batch_size:(i + 1) * self.batch_size]

        results_g = self.sess.run(self.G, feed_dict={self.z: batch_data})
        results_d = self.sess.run(self.D_logits, feed_dict={self.inputs: batch_data})
        #results = self.sess.run(self.sampler, feed_dict={self.z: batch_data})

        # to log some images with d values
        #for idx,image in enumerate(results_g):
        #  scipy.misc.imsave('samples/{}_{}.jpg'.format(idx,results_d[idx][0]),batch_data[idx,:,:,0])

        lst_discriminator_v.extend(results_d)
        lst_generated_img.extend(results_g)
        print('Tested batch {}/{}'.format(i+1,batch_idxs))

    #f = plt.figure()
    #plt.plot(np.array(lst_discriminator_v))
    #f.savefig('samples/d_values.jpg')

    scipy.misc.imsave('./'+self.sample_dir+'/ALOCC_generated.jpg', montage(np.array(lst_generated_img)[:,:,:,0]))
    scipy.misc.imsave('./'+self.sample_dir+'/ALOCC_input.jpg', montage(np.array(tmp_lst_slices)[:,:,:,0]))
    
    return lst_discriminator_v
