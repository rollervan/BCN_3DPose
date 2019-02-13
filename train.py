# coding=utf-8
import tensorflow as tf
import numpy as np

import cv2
import os, shutil, re
import math

from random import shuffle
from scipy.misc import imresize
import scipy.io as io

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D


class BCN:

    def __init__(self, restore):
        self.global_path = './'
        self.data_dir = '/home/ivan/TF/Data3D/'

        self.restore = restore
        if self.restore:
            self.model_name = 'BCN_3D'
            self.checkpoint = self.global_path+'Models/Model_'+self.model_name+'/model.ckpt'
            self.log_dir = self.global_path+'/logs/'+self.model_name
            if not os.path.exists(self.global_path + 'temp'):
                os.makedirs(self.global_path + 'temp')

        else:
            self.model_name = 'BCN_3D'
            if not os.path.exists(self.global_path+'Models/Model_'+self.model_name):
                os.makedirs(self.global_path+'Models/Model_'+self.model_name)
            self.checkpoint = self.global_path+'Models/Model_'+self.model_name+'/model.ckpt'

            self.log_dir = self.global_path+'/logs/'+self.model_name
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

        #Training hyperparameters

        self.learning_rate = 1e-5
        self.IM_ROWS = 256
        self.IM_COLS = 256
        self.IM_DEPTH = 3
        self.eps = 1e-9
        self.batch_size = 20
        self.epoch = 50
        self.training_examples = 311724

        #CapsNet parameters

        self.num_articulaciones = 17
        self.vec_len = 8
        self.out_vec_len = 16
        self.iter_routing = 3
        self.num_outputs = 17  # número de conceptos
        self.stddev = 0.01

    def procrustes(self, X, Y, scaling=True, reflection='best'):
        """
        A port of MATLAB's `procrustes` function to Numpy.

        Procrustes analysis determines a linear transformation (translation,
        reflection, orthogonal rotation and scaling) of the points in Y to best
        conform them to the points in matrix X, using the sum of squared errors
        as the goodness of fit criterion.

            d, Z, [tform] = procrustes(X, Y)

        Inputs:
        ------------
        X, Y
            matrices of target and input coordinates. they must have equal
            numbers of  points (rows), but Y may have fewer dimensions
            (columns) than X.

        scaling
            if False, the scaling component of the transformation is forced
            to 1

        reflection
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.

        Outputs
        ------------
        d
            the residual sum of squared errors, normalized according to a
            measure of the scale of X, ((X - X.mean(0))**2).sum()

        Z
            the matrix of transformed Y-values

        tform
            a dict specifying the rotation, translation and scaling that
            maps X --> Y

        """

        n, m = X.shape
        ny, my = Y.shape

        muX = X.mean(0)
        muY = Y.mean(0)

        X0 = X - muX
        Y0 = Y - muY

        ssX = (X0 ** 2.).sum()
        ssY = (Y0 ** 2.).sum()

        # centred Frobenius norm
        normX = np.sqrt(ssX)
        normY = np.sqrt(ssY)

        # scale to equal (unit) norm
        X0 /= normX
        Y0 /= normY

        if my < m:
            Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

        # optimum rotation matrix of Y
        A = np.dot(X0.T, Y0)
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt.T
        T = np.dot(V, U.T)

        if reflection is not 'best':

            # does the current solution use a reflection?
            have_reflection = np.linalg.det(T) < 0

            # if that's not what was specified, force another reflection
            if reflection != have_reflection:
                V[:, -1] *= -1
                s[-1] *= -1
                T = np.dot(V, U.T)

        traceTA = s.sum()

        if scaling:

            # optimum scaling of Y
            b = traceTA * normX / normY

            # standarised distance between X and b*Y*T + c
            d = 1 - traceTA ** 2

            # transformed coords
            Z = normX * traceTA * np.dot(Y0, T) + muX

        else:
            b = 1
            d = 1 + ssY / ssX - 2 * traceTA * normY / normX
            Z = normY * np.dot(Y0, T) + muX

        # transformation matrix
        if my < m:
            T = T[:my, :]
        c = muX - b * np.dot(muY, T)

        # transformation values
        tform = {'rotation': T, 'scale': b, 'translation': c}

        return d, Z, tform
    def rotate(self,xyz):
        def dotproduct(v1, v2):
            return sum((a * b) for a, b in zip(v1, v2))

        def length(v):
            return math.sqrt(dotproduct(v, v))

        def angle(v1, v2):
            num = dotproduct(v1, v2)
            den = (length(v1) * length(v2))
            if den == 0:
                print('den = 0')
                print(length(v1))
                print(length(v2))
                print(num)
            ratio = num/den
            ratio = np.minimum(1,ratio)
            ratio = np.maximum(-1,ratio)

            return math.acos(ratio)

        p1 = np.float32(xyz[1, :])
        p2 = np.float32(xyz[6, :])
        v1 = np.subtract(p2, p1)
        mod_v1 = np.sqrt(np.sum(v1 ** 2))
        x = np.float32([1., 0., 0.])
        y = np.float32([0., 1., 0.])
        z = np.float32([0., 0., 1.])
        theta = math.acos(np.sum(v1 * z) / (mod_v1 * 1)) * 360 / (2 * math.pi)
        # M = cv2.getAffineTransform()
        p = np.cross(v1, z)
        # if sum(p)==0:
        #     p = np.cross(v1,y)
        p[2] = 0.
        # ang = -np.minimum(np.abs(angle(p, x)), 2 * math.pi - np.abs(angle(p, x)))
        ang = angle(x, p)

        if p[1] < 0:
            ang = -ang

        M = [[np.cos(ang), np.sin(ang), 0.], [-np.sin(ang), np.cos(ang), 0.], [0., 0., 1.]]
        M = np.reshape(M, [3, 3])
        xyz = np.transpose(xyz)
        xyz_ = np.matmul(M, xyz)
        xyz_ = np.transpose(xyz_)


        return xyz_

    def flip_3d(self,msk):
        msk[:,1] = -msk[:,1]
        return msk
    def remove_logs(self,folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def create_list(self,data_path):
        if os.path.isfile(self.global_path+'train_files.npy'):
            train_files = np.load('train_files.npy')
            validation_files = np.load('validation_files.npy')
            print('Files have been loaded')
        else:
            train_files = [f for f in os.listdir(data_path+'train') if not re.search(r'.npy', f)]
            np.save('train_files',train_files)
            validation_files = [f for f in os.listdir(data_path+'validation') if not re.search(r'.npy', f)]
            np.save('validation_files',validation_files)
            train_files = np.load('train_files.npy')
            validation_files = np.load('validation_files.npy')
            print('Files have been created')

        self.num_images = len(train_files)
        print('Loaded Training samples: ' + str(self.num_images))
        self.training_iters = np.round(self.epoch * self.num_images / (self.batch_size)).astype(int)
        print('Training iterations: ' + str(self.training_iters))

        num_train_examples = len(train_files)
        train_indices = list(range(num_train_examples))
        num_validation_examples = len(validation_files)
        validation_indices = list(range(num_validation_examples))

        shuffle(train_indices)
        shuffle(validation_indices)

        t_f = train_files[train_indices]
        v_f = validation_files[validation_indices]


        return t_f, v_f
    def load_data_train(self,data_path, input_list):

        num_images = len(input_list)
        list_images = input_list
        list_masks = []
        for f in list_images:
            name = f[:-4] + '.npy'
            list_masks.append(name)

        print('Number of training images: ' + str(num_images))
        art_select_ = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

        while True:
            c = list(zip(list_images, list_masks))

            shuffle(c)

            list_images, list_masks = zip(*c)
            print('Training list has been shuflled')
            idx = -1
            for i in range(int(np.ceil(num_images / self.batch_size))):

                im_ = []
                msk_ = []
                rm_ = []
                msk2d_ = []
                for d in range(self.batch_size):

                    if idx >= num_images - 1:
                        idx = idx + 1 - num_images
                    else:
                        idx = idx + 1

                    gt17 = data_path + '_bg17/' + list_images[idx]
                    gt17 = gt17[:-4] + '.npz'
                    loaded = np.load(gt17)
                    rm = loaded['gt']

                    im = cv2.imread(data_path + '/' + list_images[idx])
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                    flip = np.random.randint(0, 2)
                    if flip:
                        im = cv2.flip(im, 1)
                        rm = cv2.flip(rm, 1)
                        # Permute
                        rm = rm[:, :, [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]]

                    rm_.append(rm)
                    im_.append(im)

                    msk = np.float32(np.load(data_path + '/' + list_masks[idx]))
                    bias = np.repeat(np.reshape(msk[0, :], [1, 3]), 32, axis=0)
                    msk = msk - bias
                    msk = self.rotate(msk)
                    msk = msk[art_select_, :]

                    msk2d = np.float32(np.load(data_path + '2d' + '/' + list_masks[idx]))
                    msk2d = msk2d[art_select_, :]


                    if flip:
                        msk = self.flip_3d(msk)
                        msk = msk[[0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13], :]
                        msk2d[:, 0] = 1. - msk2d[:, 0]
                        msk2d = msk2d[[0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13],:]

                    msk_.append(msk)
                    msk2d_.append(msk2d)

                im = np.float32(np.reshape(im_, [self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH]) / 255.)
                msk = np.reshape(msk_, [self.batch_size, self.num_articulaciones, 3]) / 1000.
                rm = np.float32(np.reshape(rm_, [self.batch_size, self.IM_ROWS, self.IM_COLS, 17]) / 1.)
                msk2d = np.reshape(msk2d_, [self.batch_size, self.num_articulaciones, 2])

                yield im, msk, rm, msk2d
    def load_data_validation(self,data_path, input_list):

        num_images = len(input_list)
        list_images = input_list
        list_masks = []
        for f in list_images:
            name = f[:-4] + '.npy'
            list_masks.append(name)

        print('Number of validation images: '+str(num_images))
        art_select_ = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

        while True:
            c = list(zip(list_images, list_masks))

            # shuffle(c)

            list_images, list_masks = zip(*c)
            print('Validation list has not been shuflled')
            idx = -1
            for i in range(int(np.ceil(num_images/self.batch_size))):

                im_ = []
                msk_ = []
                rm_ = []
                msk2d_ = []
                for d in range(self.batch_size):

                    if idx >= num_images-1:
                        idx = idx + 1 - num_images
                    else:
                        idx = idx + 1

                    gt17 = data_path+'_bg17/'+ list_images[idx]
                    gt17 = gt17[:-4]+'.npz'
                    loaded = np.load(gt17)
                    rm = loaded['gt']
                    rm_.append(rm)

                    im = cv2.imread(data_path +'/'+ list_images[idx])
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

                    im_.append(im)

                    msk = np.float32(np.load(data_path + '/'+list_masks[idx]))
                    bias = np.repeat(np.reshape(msk[0, :], [1, 3]), 32, axis=0)
                    msk = msk - bias
                    msk = self.rotate(msk)
                    msk = msk[art_select_, :]
                    msk_.append(msk)

                    msk2d = np.float32(np.load(data_path + '2d' + '/' + list_masks[idx]))
                    msk2d = msk2d[art_select_,:]
                    msk2d_.append(msk2d)

                im = np.float32(np.reshape(im_,[self.batch_size,self.IM_ROWS,self.IM_COLS,self.IM_DEPTH])/255.)
                msk = np.reshape(msk_,[self.batch_size,self.num_articulaciones,3])/1000.
                rm = np.float32(np.reshape(rm_,[self.batch_size,self.IM_ROWS,self.IM_COLS,17])/1.)
                msk2d = np.reshape(msk2d_,[self.batch_size,self.num_articulaciones,2])

                yield im, msk, rm, msk2d

    def routing(self,input,num_outputs,in_dim,out_dim,is_training):

        b_IJ = tf.constant(np.zeros([1, input.shape[1].value, num_outputs, 1, 1], dtype=np.float32))

        W = tf.get_variable('Weight_W', shape=(1, self.num_pred_v , num_outputs, in_dim, out_dim), dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=self.stddev)) #stddev=self.stddev
        Winv = tf.get_variable('Weight_Winv', shape=(1, self.num_pred_v , num_outputs, in_dim, out_dim), dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=self.stddev)) #stddev=self.stddev
        input = tf.tile(input, [1, 1, num_outputs, 1, 1])
        W = tf.tile(W, [self.batch_size, 1, 1, 1, 1])
        Winv = tf.tile(Winv, [self.batch_size, 1, 1, 1, 1])
        with tf.name_scope('W_Reg'):

            w_reg =  tf.matmul(Winv,W,transpose_b=True) - tf.eye(8)
            w_reg = tf.reshape(w_reg,shape=[self.batch_size,w_reg.shape[1].value,w_reg.shape[2].value,-1])
            W_reg = tf.reduce_mean(tf.reduce_sum(tf.square(w_reg, name='Norm_W'), axis=-1))

        assert input.get_shape() == [self.batch_size, self.num_pred_v , num_outputs, in_dim, 1]

        input = tf.layers.dropout(input,rate=0.3,training=is_training)

        u_hat =  tf.matmul(W, input, transpose_a=True)
        u_hat_stop_gradient = tf.stop_gradient(u_hat,name="stop_grad")
        assert u_hat.get_shape() == [self.batch_size, self.num_pred_v , num_outputs, out_dim, 1]

        # line 3,for r iterations do
        for r_iter in range(self.iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                # line 4:
                # => [1, 1152, 10, 1, 1]
                c_IJ = tf.nn.softmax(b_IJ, axis=2)
                c_IJ = tf.tile(c_IJ, [self.batch_size, 1, 1, 1, 1])
                if r_iter == self.iter_routing - 1:
                    assert c_IJ.get_shape() == [self.batch_size, self.num_pred_v , num_outputs, 1, 1]

                    # line 5:
                    # weighting u_hat with c_IJ, element-wise in the last two dims
                    # => [batch_size, 1152, 10, 16, 1]
                    s_J = tf.multiply(c_IJ, u_hat)
                    # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                    assert s_J.get_shape() == [self.batch_size, 1, num_outputs, out_dim, 1]

                    tf.add_to_collections('cjs',c_IJ)

                    # line 6:
                    # squash using Eq.1,
                    v_J = self.squash(s_J)
                    # v_J = self.ball_proj(s_J)
                else:
                    assert c_IJ.get_shape() == [self.batch_size, self.num_pred_v , num_outputs, 1, 1]

                    # line 5:
                    # weighting u_hat with c_IJ, element-wise in the last two dims
                    # => [batch_size, 1152, 10, 16, 1]
                    s_J = tf.multiply(c_IJ, u_hat_stop_gradient)
                    # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                    assert s_J.get_shape() == [self.batch_size, 1, num_outputs, out_dim, 1]

                    # line 6:
                    # squash using Eq.1,
                    v_J = self.squash(s_J)
                    # v_J = self.ball_proj(s_J)

                    assert v_J.get_shape() == [self.batch_size, 1, num_outputs, out_dim, 1]

                    # line 7:
                    # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 10, 1152, 16, 1]
                    # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                    # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                    v_J_tiled = tf.tile(v_J, [1, self.num_pred_v , 1, 1, 1])
                    u_produce_v = tf.matmul(u_hat_stop_gradient, v_J_tiled, transpose_a=True)
                    assert u_produce_v.get_shape() == [self.batch_size, self.num_pred_v , num_outputs, 1, 1]
                    b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)

        return v_J, W_reg

    def squash(self,vector):
        '''Squashing function corresponding to Eq. 1
        Args:
            vector: A 5-D tensor with shape [batch_size, 1, num_caps, vec_len, 1],
        Returns:
            A 5-D tensor with the same shape as vector but squashed in 4rd and 5th dimensions.
        '''
        epsilon = 1e-9
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
        vec_squashed = scalar_factor * vector  # element-wise
        return (vec_squashed)

        # norm = tf.norm(vector, axis=2, keep_dims=True)
        # norm_squared = norm * norm
        # return (vector / norm) * (norm_squared / (1 + norm_squared))
    def compute_distances(self,labels3D,predictions3D,labels2D,predictions2D):
        ED_list = tf.reduce_sum(tf.pow(tf.subtract(predictions3D, labels3D), 2.), axis=2)
        # ED_list_p = tf.pow(tf.reduce_sum( tf.pow( ED_list,4.), axis=1 ),1/4)

        ED = tf.reduce_mean(ED_list)
        EDs = tf.reduce_mean(tf.sqrt(ED_list))
        # EDw = tf.reduce_mean( 1*ED_list_p )

        ED_list_2d = tf.reduce_sum(tf.pow(tf.subtract(predictions2D, labels2D), 2.), axis=2)
        # ED_list_2d_p = tf.pow(tf.reduce_sum( tf.pow( ED_list_2d,4.), axis=1 ),1/4)

        ED_2d = tf.reduce_mean(ED_list_2d)
        EDs_2d = tf.reduce_mean(tf.sqrt(ED_list_2d))
        return ED,ED_2d,EDs, EDs_2d

    def capsnet(self, input_images, is_training=False):

        with tf.variable_scope('CapsNet'):
            with tf.variable_scope('CNN'):
                #------------------------------------CONV---------------------------------------#
                out = tf.layers.conv2d(input_images, 32, [9, 9], strides=(2, 2), padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1',
                                       activation=tf.nn.relu)

                out1 = out + tf.layers.conv2d(input_images,32,[1, 1],strides=(2, 2), padding='same',name='Identity1')

                # ------------------------------------CONV---------------------------------------#

                out = tf.layers.conv2d(out1, 64, [9, 9], strides=(2, 2), padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv2',
                                       activation=tf.nn.relu)

                out2 = out + tf.layers.conv2d(out1,64,[1, 1],strides=(2, 2), padding='same',name='Identity2')

                # ------------------------------------CONV---------------------------------------#

                out = tf.layers.conv2d(out2, 128, [9, 9], strides=(2, 2), padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv3',
                                       activation=tf.nn.relu)

                out3 = out + tf.layers.conv2d(out2,128,[1, 1],strides=(2, 2), padding='same',name='Identity3')

                # ------------------------------------CONV---------------------------------------#

                out = tf.layers.conv2d(out3, 16, [9, 9], strides=(2, 2), padding='same',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv4',
                                       activation=tf.nn.relu)

                out = out + tf.layers.conv2d(out3,16,[1, 1],strides=(2, 2), padding='same',name='Identity4')


            with tf.variable_scope('Routing'):
                capsules = tf.reshape(out, (self.batch_size, -1, self.vec_len, 1))
                capsules = self.squash(capsules)
                self.num_pred_v = capsules.shape[1].value
                capsules = tf.reshape(capsules, shape=(self.batch_size, -1, 1, capsules.shape[-2].value, 1))
                capsules, W_reg = self.routing(capsules, self.num_outputs,self.vec_len,self.out_vec_len,is_training)

                capsules = tf.squeeze(capsules, axis=1)

            with tf.variable_scope('Entities'):
                resCap = tf.sqrt(tf.reduce_sum(tf.square(capsules), axis=2, keep_dims=True) + self.eps)

                flat_capsules = tf.reshape(capsules,shape=(self.batch_size,-1))

            with tf.variable_scope('FC_2D'):
                fc = tf.layers.dense(flat_capsules,1024,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None)
                fc = tf.nn.relu(fc)
                fc = tf.layers.dense(fc,2048,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)

                if 1:
                    fc = tf.layers.dropout(fc, 0.15, training=is_training)
                fc = tf.layers.dense(fc,2*self.num_articulaciones,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.sigmoid)

                res2D = tf.reshape(fc,(self.batch_size,self.num_articulaciones,2,1))

            with tf.variable_scope('FC_3D'):
                fc = tf.layers.dense(flat_capsules,1024,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=None)
                fc = tf.nn.relu(fc)
                fc = tf.layers.dense(fc,2048,kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu)

                if 1:
                    fc = tf.layers.dropout(fc,0.15,training=is_training)
                fc = tf.layers.dense(fc,3*self.num_articulaciones, kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.tanh)

                res3D = tf.reshape(fc,(self.batch_size,self.num_articulaciones,3,1))

            with tf.variable_scope('Recons'):
                fc = tf.layers.dense(flat_capsules, 1024,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)

                fc = tf.layers.dense(fc, 2048,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)

                if 1:
                    fc = tf.layers.dropout(fc,rate=0.15,training=is_training)
                fc = tf.layers.dense(fc, 64*64*self.num_articulaciones,kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)

                fc = tf.reshape(fc, (self.batch_size, 64,64, self.num_articulaciones))
                resR = tf.image.resize_images(fc,size=[self.IM_ROWS,self.IM_COLS])

        return res3D, res2D, resR, resCap, W_reg

    def train(self):

        is_training = tf.placeholder(dtype=tf.bool,name='is_training')


        images = tf.placeholder(dtype=tf.float32,shape=[self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH])

        imagess = tf.cond(is_training,lambda:tf.image.random_hue(images,max_delta=0.5), lambda: images)
        imagess = tf.cond(is_training,lambda:tf.image.random_contrast(imagess,lower=0.5, upper=1.5), lambda: images)

        msk3D = tf.placeholder(dtype=tf.float32,shape=[self.batch_size, self.num_articulaciones, 3, 1])
        msk2D = tf.placeholder(dtype=tf.float32,shape=[self.batch_size, self.num_articulaciones, 2, 1])
        mskR = tf.placeholder(dtype=tf.float32,shape=[self.batch_size, self.IM_ROWS, self.IM_COLS, 17])

        res3D, res2D, resR, resCap, W_reg = self.capsnet(input_images= imagess, is_training=is_training)

        ED, ED_2D, EDs, EDs_2D = self.compute_distances(labels3D=msk3D,predictions3D=res3D,labels2D=msk2D,predictions2D=res2D)

        loss3D = ED
        loss2D = ED_2D


        flat_mskR = tf.reshape(mskR, [self.batch_size, -1])
        flat_resR = tf.reshape(resR, [self.batch_size, -1])
        lossR = tf.reduce_mean(tf.reduce_mean(tf.square(flat_mskR - flat_resR), axis=1))


        cnn_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CapsNet')
        cnn_var_wo_bn = []
        for var in cnn_var:
            if ('conv' in var.name):
                cnn_var_wo_bn.append(var)

        with tf.name_scope('Bayesian_Weights'):
            s1 = tf.Variable(1.,name='s1')
            s2 = tf.Variable(1.,name='s2')
            s3 = tf.Variable(1.,name='s3')
            s4 = tf.Variable(1.,name='s4')

            c1 = 0.5*tf.exp(-s1,name='c1')
            c2 = 0.5*tf.exp(-s2,name='c2')
            c3 = 0.5*tf.exp(-s3,name='c3')
            c4 = 0.5*tf.exp(-s4,name='c4')


        loss = c1*loss3D + c2*loss2D + c3*lossR + c4*W_reg + s1 + s2 + s3 + s4

        saver = tf.train.Saver()

        global_step = tf.Variable(0, trainable=False)

        learning_rate = self.learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
           train_op = optimizer.minimize(loss,global_step=global_step)

        saver_all = tf.train.Saver()

        # Summaries

        with tf.name_scope('Losses'):
            tf.summary.scalar('Total_Loss', loss)
            tf.summary.scalar('Recon_Loss',lossR)
            tf.summary.scalar('loss3D',loss3D)
            tf.summary.scalar('loss2D',loss2D)
            tf.summary.scalar('W_reg',W_reg)
        with tf.name_scope('Metrics'):
            tf.summary.scalar('ED', 1000 * EDs)
            tf.summary.scalar('ED2D', 1000 * EDs_2D)
        with tf.name_scope('Coefs'):
            tf.summary.scalar('c1', c1)
            tf.summary.scalar('c2', c2)
            tf.summary.scalar('c3', c3)
            tf.summary.scalar('c4', c4)

        summary = tf.summary.merge_all()

        #Remove previous logs
        if not(self.restore):
            self.remove_logs(self.log_dir)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        with tf.Session() as sess:

            summary_writer_t = tf.summary.FileWriter(self.log_dir+'/train', sess.graph,filename_suffix='train')
            summary_writer_v = tf.summary.FileWriter(self.log_dir+'/validation', sess.graph,filename_suffix='valid')

            sess.run(tf.global_variables_initializer())

            if self.restore:
                saver.restore(sess,self.checkpoint)

            t_f, v_f = self.create_list(data_path=self.data_dir)

            data_train = self.load_data_train(data_path=self.data_dir+'train', input_list=t_f)
            data_valid = self.load_data_validation(data_path=self.data_dir+'validation', input_list=v_f)

            for i in tqdm(range(self.training_iters)):

                ti, tm, tr, tm2d = next(data_train)
                if not((i % 100) == 0):
                    ti = np.reshape(ti,[self.batch_size,self.IM_ROWS,self.IM_COLS,self.IM_DEPTH]).astype(np.float32)
                    tm = np.reshape(tm,[self.batch_size,self.num_articulaciones,3,1]).astype(np.float32)
                    tr = np.reshape(tr,[self.batch_size,self.IM_ROWS,self.IM_COLS,17]).astype(np.float32)
                    tm2d = np.reshape(tm2d,[self.batch_size,self.num_articulaciones,2,1]).astype(np.float32)

                    _ = sess.run([train_op],
                        feed_dict={images: ti, msk3D: tm, msk2D: tm2d, mskR: tr, is_training: True})


                if ((i % 100) == 0):
                    ti, tm, tr, tm2d = next(data_train)

                    ti = np.reshape(ti, [self.batch_size, self.IM_ROWS, self.IM_COLS, self.IM_DEPTH]).astype(np.float32)
                    tm = np.reshape(tm, [self.batch_size, self.num_articulaciones, 3, 1]).astype(np.float32)
                    tr = np.reshape(tr, [self.batch_size, self.IM_ROWS, self.IM_COLS, 17]).astype(np.float32)
                    tm2d = np.reshape(tm2d, [self.batch_size, self.num_articulaciones, 2, 1]).astype(np.float32)

                    _, res3D_value, res2D_value, resR_value, summary_str = sess.run(
                        [train_op,res3D, res2D, resR, summary],
                        feed_dict={images: ti, msk3D: tm, msk2D: tm2d, mskR: tr, is_training: True},
                              options=run_options,
                              run_metadata=run_metadata)

                    summary_writer_t.add_run_metadata(run_metadata,'step%d' % i)
                    summary_writer_t.add_summary(summary_str, i)
                    summary_writer_t.flush()

                    vi, vm, vr, vm2d = next(data_valid)
                    vi = np.reshape(vi,[self.batch_size,self.IM_ROWS,self.IM_COLS,self.IM_DEPTH]).astype(np.float32)
                    vm = np.reshape(vm,[self.batch_size,self.num_articulaciones,3,1]).astype(np.float32)
                    vr = np.reshape(vr,[self.batch_size,self.IM_ROWS,self.IM_COLS,17]).astype(np.float32)
                    vm2d = np.reshape(vm2d,[self.batch_size,self.num_articulaciones,2,1]).astype(np.float32)

                    res3D_v_value, res2D_v_value, resR_v_value, summary_str = sess.run([res3D, res2D, resR, summary],
                        feed_dict={images: vi, msk3D: vm, msk2D: vm2d, mskR: vr, is_training: False})


                    summary_writer_v.add_summary(summary_str, i)
                    summary_writer_v.flush()

                    if ((i % 100) == 0) and (not(i == 0)):
                        np.save(self.global_path+'temp/valid_out', res3D_v_value[0, :, :, :])
                        np.save(self.global_path+'temp/valid_gt', vm[0, :, :, :])
                        np.save(self.global_path+'temp/train_out', res3D_value[0, :, :, :])
                        np.save(self.global_path+'temp/train_gt', tm[0, :, :, :])

                        np.save(self.global_path+'temp/valid_out_2d', res2D_v_value[0, :, :, :])
                        np.save(self.global_path+'temp/valid_gt_2d', vm2d[0, :, :, :])
                        np.save(self.global_path+'temp/train_out_2d', res2D_value[0, :, :, :])
                        np.save(self.global_path+'temp/train_gt_2d', tm2d[0, :, :, :])

                        cv2.imwrite(self.global_path+'temp/valid_im.jpg', cv2.cvtColor(255 * vi[0, :, :, :], cv2.COLOR_BGR2RGB))
                        cv2.imwrite(self.global_path+'temp/train_im.jpg', cv2.cvtColor(255 * ti[0, :, :, :], cv2.COLOR_BGR2RGB))

                        np.save(self.global_path+'temp/train_recon.npy',resR_value)
                        np.save(self.global_path+'temp/valid_recon.npy', resR_v_value)
                        np.save(self.global_path+'temp/tr.npy', tr)
                        np.save(self.global_path+'temp/vr.npy', vr)

                if (i % np.round(self.training_examples/self.batch_size))==0 and not(i==0):
                    saver_all.save(sess, self.checkpoint)

                if ((i % 1000) == 0) and (not(i == 0)):
                    saver_all.save(sess, self.checkpoint)


            sess.close()

    def test(self):

        # children = io.loadmat('children_b.mat')['b']
        self.batch_size = 1
        image_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size , self.IM_ROWS, self.IM_COLS, 3])

        net_output = self.capsnet(image_placeholder,is_training=False)

        saver_cnn = tf.train.Saver()

        if os.path.isfile('validation_files.npy'):
            validation_files = np.load('validation_files.npy')
        activities = ['Directions','Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing', 'Purchases', 'Sitting','SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'Walking', 'WalkTogether']

        for activity in activities:
            i = 1
            test_list = []
            for file in validation_files:
                if activity in file:
                    if activity == 'Sitting':
                        if 'SittingDown' not in file:
                            test_list.append(file)
                            i += 1
                    else:
                        test_list.append(file)
                        i += 1
            if not os.path.exists(self.global_path+'Test_lists'):
                os.makedirs(self.global_path+'Test_lists')
            np.save('Test_lists/Test_'+activity+'.npy',test_list)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # Cargamos las variables
            saver_cnn.restore(sess, self.checkpoint)
            ED_total = []
            ED_list_avg = []

            for activity in activities:
                print(activity)
                ED_list = []
                list = np.load(self.global_path+'Test_lists/Test_'+activity+'.npy')
                for subject in list:
                    image = cv2.imread(self.data_dir+'validation/'+subject)
                    xyz_total_gt = np.load(self.data_dir+'validation/'+subject[:-4]+ '.npy')
                    art_select_ = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
                    bias = np.repeat(np.reshape(xyz_total_gt[0, :], [1, 3]), 32, axis=0)
                    xyz_total_gt = xyz_total_gt - bias
                    xyz_total_gt = self.rotate(xyz_total_gt)
                    xyz_total_gt = xyz_total_gt[art_select_,:]

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = imresize(image, [self.IM_ROWS, self.IM_COLS])

                    image = np.reshape(image, (1, self.IM_ROWS, self.IM_COLS, 3)) / 255.
                    r = sess.run([net_output], feed_dict={ image_placeholder: image})
                    r1 = r[0][0][0,:,:,:]
                    xyz_total = np.squeeze(1 * r1,axis=2)

                    procrustes_transform = 0
                    if procrustes_transform == 1:
                        d, Z, tform = self.procrustes(xyz_total_gt, 1000*xyz_total)
                        ED = np.average(np.sqrt(np.sum(np.square(xyz_total_gt-Z),axis=1)))
                    else:
                        ED = np.average(np.sqrt(np.sum(np.square(xyz_total_gt-1000*xyz_total),axis=1)))

                    ED_list.append(ED)
                    ED_total.append(ED)
                ED_list_avg.append(np.mean(ED_list))
                print(np.average(ED_list))

            print(np.mean(ED_list_avg))
            np.save(self.global_path+'Models/Model_'+self.model_name+ '/TOTAL_by_activity'+'.npy', ED_list_avg)
            np.save(self.global_path+'Models/Model_'+self.model_name+ '/TOTAL'+'.npy', ED_total)

            print('Total by activity Average: '+np.average(ED_total))
        return 0

    def predict_video(self):

        children = io.loadmat('./utils/children_b.mat')['b']
        #fig = plt.figure(num=None, figsize=(16, 7),  facecolor='w', edgecolor='k')#dpi=500,
        fig = plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')

        ax = fig.add_subplot(122, projection='3d')
        ax1 = fig.add_subplot(121)

        image_placeholder = tf.placeholder(tf.float32, shape=[1, self.IM_ROWS, self.IM_COLS, 3])
        self.batch_size = 1
        net_output = self.capsnet(image_placeholder,is_training=False)

        saver_cnn = tf.train.Saver()

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=30, metadata=metadata)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # Cargamos las variables
            saver_cnn.restore(sess, self.checkpoint)
            angle = 0
            with writer.saving(fig, "S11_Sitting.mp4", 100):

                for i in range(1,3000):
                    print(i)
                    if angle >= 360:
                        angle = 0
                    else:
                        angle = angle +1
                    angle = 0

                    image = cv2.imread(self.data_dir+'validation/S11_Sitting 1.60457274_'+str(5*i)+'.jpg')
                    xyz_total_gt = np.load(self.data_dir+'validation/S11_Sitting 1.60457274_' + str(5 * i) + '.npy')
                    art_select_ = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

                    xyz_total_gt = self.rotate(xyz_total_gt)
                    xyz_total_gt = xyz_total_gt[art_select_,:]

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = imresize(image, [self.IM_ROWS, self.IM_COLS])
                    plt.clf()
                    ax1 = fig.add_subplot(131)
                    ax1.imshow(image)

                    image = np.reshape(image, (1, self.IM_ROWS, self.IM_COLS, 3)) / 255.
                    # Evaluamos el tensor `labels` pasandole nuestra imagen de entrada `image` a la red a traves del placeholder
                    r = sess.run([net_output], feed_dict={ image_placeholder: image})
                    r1 = r[0][0][0,:,:,:]
                    bias = np.repeat(np.reshape(r1[0, :,0], [1, 3]), self.num_articulaciones, axis=0)

                    r2 = r[0][1][0,:,:,:]

                    xyz_total = 1000 * r1
                    xy_total = 1000 * r2
                    xy_total[:, 1, :] = -xy_total[:, 1, :]
                    ax = fig.add_subplot(132, projection='3d')


                    for i in range(len(children)):
                        child = children[i]
                        child = child[0]

                        if child.size:
                            child = child[0]
                            for j in range(len(child)):
                                seg = []
                                P2 = xyz_total[i, :]
                                P1 = xyz_total[child[j] - 1, :]
                                ax.plot([P1[0], P2[0]], [P1[1], P2[1]], zs=[P1[2], P2[2]])
                        # ax.hold(True)

                    # ax.scatter(xyz_total[:, 0], xyz_total[:, 1], xyz_total[:, 2], s=20, c=None, depthshade=True)
                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')
                    ax.set_xlim([-700, 700])
                    ax.set_ylim([-700, 700])
                    ax.set_zlim([-700, 700])
                    ax.view_init(30, angle)


                    ax2 = fig.add_subplot(133, projection='3d')

                    xyz_total_gt =  np.reshape(xyz_total_gt,(self.num_articulaciones,3,1))
                    bias = np.repeat(np.reshape(xyz_total_gt[0, :,0], [1, 3]), self.num_articulaciones, axis=0)
                    xyz_total_gt = xyz_total_gt - np.reshape(bias,(self.num_articulaciones,3,1))
                    for i in range(len(children)):
                        child = children[i]
                        child = child[0]

                        if child.size:
                            child = child[0]
                            for j in range(len(child)):
                                seg = []
                                P2 = xyz_total_gt[i, :]
                                P1 = xyz_total_gt[child[j] - 1, :]
                                ax2.plot([P1[0], P2[0]], [P1[1], P2[1]], zs=[P1[2], P2[2]])

                    # ax1.scatter(xyz_total_gt[:, 0], xyz_total_gt[:, 1], xyz_total_gt[:, 2], s=3, c=None, depthshade=True)
                    ax2.set_xlabel('X Label')
                    ax2.set_ylabel('Y Label')
                    ax2.set_zlabel('Z Label')
                    ax2.set_xlim([-700, 700])
                    ax2.set_ylim([-700, 700])
                    ax2.set_zlim([-700, 700])
                    ax2.view_init(30, angle)


                    plt.draw()
                    plt.pause(0.01)
                        # x0 += 0.1 * np.random.randn()
                        # y0 += 0.1 * np.random.randn()
                        # l.set_data(x0, y0)
                    writer.grab_frame()
                    # Nos quedamos solo con el primer canal y quitamos las dos dimensiones que no nos interesan (batch y canales) usando reshape
        sess.close()
        return 0

