# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:59:54 2018
使用Y作为上采样的先验， 使用共同的B， 并输出所有层的X
@author: XieQi
"""
#import h5py
import os
import skimage.measure
import numpy as np
import scipy.io as sio    
import re
import CAVE_dataReader as Crd
import tensorflow as tf
import MyLib as ML
import random 
import HSInet_DownSam_allratio_shareC as Hn
slim = tf.contrib.slim

# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS

# 模式：训练、测试
tf.app.flags.DEFINE_string('mode', 'testAllGAN', 
                           'train or trainGAN or test or iniTest or testAll or testAllGAN.')
# 训练GAN时生成器是从头开始训练还是用已经训练好的参数
tf.app.flags.DEFINE_string('gentrain', 'retrain',
                           'restore or retrain')
# 定义gan损失的权重
tf.app.flags.DEFINE_float('lam3', 0.0005,
                            'lambda')

# 网络结构：resnet， CRNN
tf.app.flags.DEFINE_integer('structure', 1,
                            'network structures.(0 for resnet and 1 for CRNN)')
# 输出的维数
tf.app.flags.DEFINE_integer('outDim', 31,
                           'output channel number') 

# lambda
tf.app.flags.DEFINE_float('lam1', 0.1,
                           'lambda') 
# lambda
tf.app.flags.DEFINE_float('lam2', 0.01,
                           'lambda') 
# HSI网络的层数(13)
tf.app.flags.DEFINE_integer('HSInetL', 6,
                           'layer number of HSInet') 
# 子网络的层数(2)
tf.app.flags.DEFINE_integer('subnetL', 2,
                           'layer number of subnet') 
# 学习率
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                           'learning_rate') 
# epoch
tf.app.flags.DEFINE_integer('epoch', 50,
                           'epoch') 
# 可视化测试数据名
tf.app.flags.DEFINE_string('train_data_path', 'h5data/',
                           'Filepattern for training data.')
# 测试数据路劲
tf.app.flags.DEFINE_string('test_data_name', 'chart_and_stuffed_toy_ms',
                           'Filepattern for eval data') 
# 训练过程数据的存放路径
tf.app.flags.DEFINE_string('train_dir', 'temp/HSICRNNnetL_6_subnetL2_ratio2_epoch25/',
                           'Directory to keep training outputs.')
# 测试过程数据的存放路径
tf.app.flags.DEFINE_string('test_dir', 'TestResult/HSICRNNnetL_6_subnetL2_ratio2_epoch25/',
                           'Directory to keep eval outputs.')
# GAN训练过程数据的存放路径
tf.app.flags.DEFINE_string('GANtrain_dir', 'temp/HSICRNNnet_GANspectral_sigmoid_lam3_00001_L_6_subnetL2_ratio2_epoch50/',
                           'Directory to keep training outputs.')
# GAN测试过程数据的存放路径
tf.app.flags.DEFINE_string('GANtest_dir', 'TestResult/HSICRNNnet_GANspectral_sigmoid_lam3_00001_L_6_subnetL2_ratio2_epoch50/',
                           'Directory to keep eval outputs.')
# 数据参数
tf.app.flags.DEFINE_integer('image_size', 96, 
                            'Image side length.')
tf.app.flags.DEFINE_integer('BatchIter', 2000,
                            """number of training h5 files.""")
#tf.app.flags.DEFINE_integer('num_patches', 200,
#                            """number of patches in each h5 file.""")
tf.app.flags.DEFINE_integer('batch_size', 10,
                            """Batch size.""")
# GPU设备数量（0代表CPU）
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')


#==============================================================================#
#test
def test(ini):
    data = sio.loadmat("CAVEdata/Z/"+FLAGS.test_data_name)
    Z    = data['Zmsi']
    data = sio.loadmat("CAVEdata/X/"+FLAGS.test_data_name)
    X    = data['msi']   
    
    
    
    ## 变为4D张量 banchsize H W C
    
    inZ = np.expand_dims(Z, axis = 0)
    inZ = tf.to_float(inZ);
    
    inX = np.expand_dims(X, axis = 0)
    inX = tf.to_float(inX);
    
    
    iniData2= sio.loadmat("CAVEdata/iniUp")
    iniUp3x3 = iniData2['iniUp1']
    
    
    ## 进入网络
    if FLAGS.structure == 0:
        outX, X1, E = Hn.HSISRnet(inZ, iniUp3x3, FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    else:
        outX, X1, E = Hn.HSISRCRNNnet(inZ, iniUp3x3, FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    
    if ini==0:
        config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver(max_to_keep = 5)
        save_path = FLAGS.train_dir
        
        with tf.Session(config=config) as sess:        
           ckpt = tf.train.latest_checkpoint(save_path)
           saver.restore(sess, ckpt) 
    #       sess.run(tf.global_variables_initializer())
           pred_X,ListX, inX = sess.run([outX, X1, inX])  
    else:
        config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
        with tf.Session(config=config) as sess:        
           sess.run(tf.global_variables_initializer())
           pred_X,ListX, inX = sess.run([outX, X1,inX])       
        
    #    print(outX.shape)
#    print(X1.shape)
#    print(inX.shape)
    ## 输出数据
    toshow  = np.hstack((ML.get3band_of_tensor(ListX[int(FLAGS.HSInetL-2)]),ML.get3band_of_tensor(pred_X)))
    toshow2 = np.hstack((ML.get3band_of_tensor(inX),ML.normalized(ML.get3band_of_tensor(pred_X))))
    toshow  = np.vstack((toshow,toshow2))
    ML.imshow(toshow)
    ML.imwrite(toshow)
    psnr = skimage.measure.compare_psnr(inX, pred_X)
    print('psnr=%.4f',psnr)
    ssim = skimage.measure.compare_ssim(inX[0,:,:,:], pred_X[0,:,:,:], multichannel=True)
    print('psnr=%.4f, ssim=%.4f',psnr,ssim)

    #==============================================================================#
#test GAN
def testGAN(ini):
    data = sio.loadmat("CAVEdata/Z/"+FLAGS.test_data_name)
    Z    = data['Zmsi']
    data = sio.loadmat("CAVEdata/X/"+FLAGS.test_data_name)
    X    = data['msi']   
    
    
    
    ## 变为4D张量 banchsize H W C
    
    inZ = np.expand_dims(Z, axis = 0)
    inZ = tf.to_float(inZ);
    
    inX = np.expand_dims(X, axis = 0)
    inX = tf.to_float(inX);
    
    
    iniData2= sio.loadmat("CAVEdata/iniUp")
    iniUp3x3 = iniData2['iniUp1']
    
    
    ## 进入网络
    if FLAGS.structure == 0:
        outX, X1, E = Hn.HSISRnet(inZ, iniUp3x3, FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    else:
        outX, X1, E = Hn.HSISRCRNNnet(inZ, iniUp3x3, FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    
    if ini==0:
        config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver(max_to_keep = 5)
        save_path = FLAGS.GANtrain_dir
        
        with tf.Session(config=config) as sess:        
           ckpt = tf.train.latest_checkpoint(save_path)
           saver.restore(sess, ckpt) 
    #       sess.run(tf.global_variables_initializer())
           pred_X,ListX, inX = sess.run([outX, X1, inX])  
    else:
        config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
        with tf.Session(config=config) as sess:        
           sess.run(tf.global_variables_initializer())
           pred_X,ListX, inX = sess.run([outX, X1,inX])       
        
    #    print(outX.shape)
#    print(X1.shape)
#    print(inX.shape)
    ## 输出数据
    toshow  = np.hstack((ML.get3band_of_tensor(ListX[int(FLAGS.HSInetL-2)]),ML.get3band_of_tensor(pred_X)))
    toshow2 = np.hstack((ML.get3band_of_tensor(inX),ML.normalized(ML.get3band_of_tensor(pred_X))))
    toshow  = np.vstack((toshow,toshow2))
    ML.imshow(toshow)
    ML.imwrite(toshow)
    psnr = skimage.measure.compare_psnr(inX, pred_X)
    print('psnr=%.4f',psnr)
    ssim = skimage.measure.compare_ssim(inX[0,:,:,:], pred_X[0,:,:,:], multichannel=True)
    print('psnr=%.4f, ssim=%.4f',psnr,ssim)
    
    
#==============================================================================#
#train
def train():
    random.seed( 1 )        
    ## 变为4D张量 banchsize H W C

    iniData2= sio.loadmat("CAVEdata/iniUp")
    iniUp3x3 = iniData2['iniUp1']
                
    X       = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.outDim))  # supervised label (None,64,64,3)
    Z       = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size/2, FLAGS.image_size/2, FLAGS.outDim)) # supervised detail layer (None,64,64,3)

    if FLAGS.structure == 0:
        outX, ListX, E = Hn.HSISRnet(Z, iniUp3x3, FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    else:
        outX, ListX, E = Hn.HSISRCRNNnet(Z, iniUp3x3, FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    
    # loss function
    loss    = tf.reduce_mean(tf.square(X - outX)) + FLAGS.lam2*tf.reduce_mean(tf.square(E))  # supervised MSE loss
    for i in range(FLAGS.HSInetL-1):
        loss = loss + FLAGS.lam1*tf.reduce_mean(tf.square(X - ListX[i]))
    
    
    lr_ = FLAGS.learning_rate
    lr  = tf.placeholder(tf.float32 ,shape = [])
    g_optim =  tf.train.AdamOptimizer(lr).minimize(loss) # Optimization method: Adam # 在这里告诉 TensorFolw 要优化的是谁？
 
    
    # 固定格式
    saver = tf.train.Saver(max_to_keep = 5)
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)    
    save_path = FLAGS.train_dir
    ML.mkdir(save_path)
    epoch = int(FLAGS.epoch)
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if tf.train.get_checkpoint_state('./model/test-model'):   # load previous trained model
            ckpt = tf.train.latest_checkpoint('./model/')
            saver.restore(sess, ckpt)
            ckpt_num = re.findall(r"\d",ckpt)
            if len(ckpt_num)==3:
                start_point = 100*int(ckpt_num[0])+10*int(ckpt_num[1])+int(ckpt_num[2])
            elif len(ckpt_num)==2:
                start_point = 10*int(ckpt_num[0])+int(ckpt_num[1])
            else:
                start_point = int(ckpt_num[0])
            print("Load success") 

        else:
            print("re-training")
            start_point = 0
        
        allX = Crd.all_train_data_in()
        
        psnrall = []
        ssimall = []  
        lossall = []          
        for j in range(start_point,epoch):   # epoch 1 to 120X[:,idx] 

#            if j+1 >(epoch/3):  # reduce learning rate
#                lr_ = FLAGS.learning_rate*0.1
            if j+1 >(4*epoch/5):
                lr_ = FLAGS.learning_rate*0.1

            Training_Loss = 0                        
            for num in range(FLAGS.BatchIter):    # h5 files 每个数据传进来搞一搞，算一下R，但是因为每个数据的batch有很多重叠，所以似乎有很多重复计算
                batch_X, batch_Z = Crd.train_data_in(allX, FLAGS.image_size, FLAGS.batch_size)

                _,lossvalue = sess.run([g_optim,loss], feed_dict={X:batch_X,Z:batch_Z,lr:lr_})  # 这应该是核心的一步
                
                Training_Loss += lossvalue  # training loss
                


                _,ifshow = divmod(num+1,200) 
                if ifshow ==1:
                    pred_X,pred_ListX = sess.run([outX, ListX], feed_dict={Z:batch_Z})
                    psnr = skimage.measure.compare_psnr(batch_X,pred_X)
                    ssim = skimage.measure.compare_ssim(batch_X,pred_X,multichannel=True)
                    CurLoss = Training_Loss/(num+1)
                    model_name = 'model-epoch'   # save model
                    print('...Training with the %d-th banch ....'%(num+1))  
                    print ('.. %d epoch training, learning rate = %.8f, Training_Loss = %.4f, PSNR = %.4f, SSIM = %.4f..'
                           %(j+1, lr_, CurLoss,  psnr, ssim))
                    psnrall.append(psnr)
                    ssimall.append(ssim)
                    lossall.append(CurLoss)
                    showX = ML.get3band_of_tensor(batch_X,nbanch=0, nframe=[0,15,30])
                    maxS = np.max(showX)
                    minS = np.min(showX)
                    toshow  = np.hstack((ML.setRange(ML.get3band_of_tensor(pred_ListX[FLAGS.HSInetL-2],nbanch=0, nframe=[0,15,30]), maxS, minS),
                                         ML.setRange(ML.get3band_of_tensor(pred_X,nbanch=0, nframe=[0,15,30]), maxS, minS)))
                    ML.imshow(toshow)
#                    ML.imwrite(toshow,('tempIm_train/epoch%d_num%d.png'%(j+1,num+1)))
    
            
            print ('The %d epoch is finished, learning rate = %.8f, Training_Loss = %.4f, PSNR = %.4f, SSIM = %.4f.' %
                  (j+1, lr_, CurLoss, psnr, ssim))
            ML.imshow(toshow)
            saver.save(sess, save_path+'model.cpkt', global_step=j)
            print('=========================================')     
            print('*****************************************')
    print('psnrall=:')
    print(psnrall)
    print('ssimall=:')
    print(ssimall)
    print('lossall=:')
    print(lossall)


#==============================================================================#
#train GAN
def trainGAN():
    random.seed( 1 )        
    ## 变为4D张量 banchsize H W C

    iniData2= sio.loadmat("CAVEdata/iniUp")
    iniUp3x3 = iniData2['iniUp1']
                
    X       = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.outDim))  # supervised label (None,64,64,3)
    Z       = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size/2, FLAGS.image_size/2, FLAGS.outDim)) # supervised detail layer (None,64,64,3)

    # 构建生成器
    if FLAGS.structure == 0:
        outX, ListX, E = Hn.HSISRnet(Z, iniUp3x3, FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    else:
        outX, ListX, E = Hn.HSISRCRNNnet(Z, iniUp3x3, FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    gen_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    dbatch = tf.concat([X, outX], 0)
    # 构建判别器
    disc = Hn.Discriminator_spectral(dbatch)
    D_x, D_G_z = tf.split(tf.squeeze(disc),2)
    
    # loss function
    if FLAGS.gentrain == 'restore':
        adv_loss = tf.reduce_mean(tf.square(D_G_z - 1.0))
        gen_loss = adv_loss
        disc_loss = (tf.reduce_mean(tf.square(D_x-1.0) + tf.square(D_G_z)))
    else:
        adv_loss = tf.reduce_mean(tf.square(D_G_z - 1.0))
        recon_loss    = tf.reduce_mean(tf.square(X - outX)) + FLAGS.lam2*tf.reduce_mean(tf.square(E))  # supervised MSE loss
        for i in range(FLAGS.HSInetL-1):
            recon_loss = recon_loss + FLAGS.lam1*tf.reduce_mean(tf.square(X - ListX[i]))
        gen_loss = FLAGS.lam3 * adv_loss + recon_loss
        disc_loss = (tf.reduce_mean(tf.square(D_x-1.0) + tf.square(D_G_z)))
    
    disc_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for x in gen_var_list:
        disc_var_list.remove(x)
        
    lr_ = FLAGS.learning_rate
    lr  = tf.placeholder(tf.float32 ,shape = [])
    gen_train =  tf.train.AdamOptimizer(lr).minimize(gen_loss, var_list=gen_var_list) # Optimization method: Adam # 在这里告诉 TensorFolw 要优化的是谁？
    disc_train =  tf.train.AdamOptimizer(lr).minimize(disc_loss, var_list=disc_var_list) 
 
    
    # 固定格式
    saver = tf.train.Saver(max_to_keep = 5)
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)    
    save_path = FLAGS.GANtrain_dir
    ML.mkdir(save_path)
    epoch = int(FLAGS.epoch)
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if tf.train.get_checkpoint_state('./model/test-model'):   # load previous trained model
            ckpt = tf.train.latest_checkpoint('./model/')
            saver.restore(sess, ckpt)
            ckpt_num = re.findall(r"\d",ckpt)
            if len(ckpt_num)==3:
                start_point = 100*int(ckpt_num[0])+10*int(ckpt_num[1])+int(ckpt_num[2])
            elif len(ckpt_num)==2:
                start_point = 10*int(ckpt_num[0])+int(ckpt_num[1])
            else:
                start_point = int(ckpt_num[0])
            print("Load success") 

        else:
            print("re-training")
            start_point = 0
        
        if FLAGS.gentrain == 'restore':
            genloader = tf.train.Saver(var_list=gen_var_list)
            genloader.restore(sess, tf.train.latest_checkpoint(FLAGS.train_dir))
            
        allX = Crd.all_train_data_in()
        
        psnrall = []
        ssimall = []  
        lossall_gen = []  
        lossall_disc = []            
        for j in range(start_point,epoch):   # epoch 1 to 120X[:,idx] 

#            if j+1 >(epoch/3):  # reduce learning rate
#                lr_ = FLAGS.learning_rate*0.1
            if j+1 >(4*epoch/5):
                lr_ = FLAGS.learning_rate*0.1

            Training_Loss_gen = 0
            Training_Loss_disc = 0
            for num in range(FLAGS.BatchIter):    # h5 files 每个数据传进来搞一搞，算一下R，但是因为每个数据的batch有很多重叠，所以似乎有很多重复计算
                batch_X, batch_Z = Crd.train_data_in(allX, FLAGS.image_size, FLAGS.batch_size)

                _,genlossvalue = sess.run([gen_train, gen_loss], feed_dict={X:batch_X,Z:batch_Z,lr:lr_})  # 这应该是核心的一步
                _,disclossvalue = sess.run([disc_train, disc_loss], feed_dict={X:batch_X,Z:batch_Z,lr:lr_})  # 这应该是核心的一步
                
                Training_Loss_gen += genlossvalue  # training loss
                Training_Loss_disc += disclossvalue  # training loss
                


                _,ifshow = divmod(num+1,200) 
                if ifshow ==1:
                    pred_X,pred_ListX = sess.run([outX, ListX], feed_dict={Z:batch_Z})
                    psnr = skimage.measure.compare_psnr(batch_X,pred_X)
                    ssim = skimage.measure.compare_ssim(batch_X,pred_X,multichannel=True)
                    CurLoss_gen = Training_Loss_gen/(num+1)
                    CurLoss_disc = Training_Loss_disc/(num+1)
                    model_name = 'model-epoch'   # save model
                    print('...Training with the %d-th batch ....'%(num+1))  
                    print ('.. %d epoch training, learning rate = %.8f, Training_Loss_gen = %.4f, Training_Loss_disc = %.4f, PSNR = %.4f, SSIM = %.4f..'
                           %(j+1, lr_, CurLoss_gen, CurLoss_disc, psnr, ssim))
                    psnrall.append(psnr)
                    ssimall.append(ssim)
                    lossall_gen.append(CurLoss_gen)
                    lossall_disc.append(CurLoss_disc)
                    showX = ML.get3band_of_tensor(batch_X,nbanch=0, nframe=[0,15,30])
                    maxS = np.max(showX)
                    minS = np.min(showX)
                    toshow  = np.hstack((ML.setRange(ML.get3band_of_tensor(batch_X,nbanch=0, nframe=[0,15,30]), maxS, minS),
                                         ML.setRange(ML.get3band_of_tensor(pred_X,nbanch=0, nframe=[0,15,30]), maxS, minS)))
                    ML.imshow(toshow)
#                    ML.imwrite(toshow,('tempIm_train/epoch%d_num%d.png'%(j+1,num+1)))
    
            
            print ('The %d epoch is finished, learning rate = %.8f, Training_Loss_gen = %.4f, Training_Loss_disc = %.4f, PSNR = %.4f, SSIM = %.4f.' %
                  (j+1, lr_, CurLoss_gen, CurLoss_disc, psnr, ssim))
            ML.imshow(toshow)
            saver.save(sess, save_path+'model.cpkt', global_step=j)
            print('=========================================')     
            print('*****************************************')
    print('psnrall=:')
    print(psnrall)
    print('ssimall=:')
    print(ssimall)
    print('lossall_gen=:')
    print(lossall_gen)
    print('lossall_disc=:')
    print(lossall_disc)
                              
#==============================================================================#

def testAll():
    ## 变为4D张量 banchsize H W C
    iniData2= sio.loadmat("CAVEdata/iniUp");
    iniUp3x3 = iniData2['iniUp1'];
    List = sio.loadmat('CAVEdata/List')
    Ind  = List['Ind']
    Z       = tf.placeholder(tf.float32, shape=(1, 512/2, 512/2, FLAGS.outDim))

    if FLAGS.structure == 0:
        outX, X1, _ = Hn.HSISRnet(Z, iniUp3x3, FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    else:
        outX, X1, _ = Hn.HSISRCRNNnet(Z, iniUp3x3, FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep = 5)
    save_path = FLAGS.train_dir
    ML.mkdir(FLAGS.test_dir)
    with tf.Session(config=config) as sess:        
        ckpt = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, ckpt) 
        for root, dirs, files in os.walk('CAVEdata/X/'):
            files.sort()
            for j in range(12): 
                i = Ind[0,j+20]-1
                data = sio.loadmat("CAVEdata/Z/"+files[i])
                inZ  = data['Zmsi']
                inZ  = np.expand_dims(inZ, axis = 0)
                inZ = tf.to_float(inZ);
                inZ = sess.run(inZ)
#                print(inZ.shape)
                pred_X,ListX = sess.run([outX, X1],feed_dict={Z:inZ})  
                pred_Lr = ListX[FLAGS.HSInetL-2]
                sio.savemat(FLAGS.test_dir+files[i], {'outX': pred_X,'outLR': pred_Lr}) 
                data = sio.loadmat("CAVEdata/X/"+files[i])
                inX    = data['msi']   
                inX  = np.expand_dims(inX, axis = 0)
                inX = tf.to_float(inX);
                inX = sess.run(inX)
                psnr = skimage.measure.compare_psnr(inX, pred_X)
                print(files[i] + ' done!'+' psnr=%.4f', psnr)

#==============================================================================#

def testAllGAN():
    ## 变为4D张量 banchsize H W C
    iniData2= sio.loadmat("CAVEdata/iniUp");
    iniUp3x3 = iniData2['iniUp1'];
    List = sio.loadmat('CAVEdata/List')
    Ind  = List['Ind']
    Z       = tf.placeholder(tf.float32, shape=(1, 512/2, 512/2, FLAGS.outDim))

    if FLAGS.structure == 0:
        outX, X1, _ = Hn.HSISRnet(Z, iniUp3x3, FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)
    else:
        outX, X1, _ = Hn.HSISRCRNNnet(Z, iniUp3x3, FLAGS.outDim,FLAGS.HSInetL,FLAGS.subnetL)

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep = 5)
    save_path = FLAGS.GANtrain_dir
    ML.mkdir(FLAGS.GANtest_dir)
    with tf.Session(config=config) as sess:        
        ckpt = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, ckpt) 
        for root, dirs, files in os.walk('CAVEdata/X/'):
            files.sort()
            for j in range(12): 
                i = Ind[0,j+20]-1
                data = sio.loadmat("CAVEdata/Z/"+files[i])
                inZ  = data['Zmsi']
                inZ  = np.expand_dims(inZ, axis = 0)
                inZ = tf.to_float(inZ);
                inZ = sess.run(inZ)
#                print(inZ.shape)
                pred_X,ListX = sess.run([outX, X1],feed_dict={Z:inZ})  
                pred_Lr = ListX[FLAGS.HSInetL-2]
                sio.savemat(FLAGS.GANtest_dir+files[i], {'outX': pred_X,'outLR': pred_Lr}) 
                data = sio.loadmat("CAVEdata/X/"+files[i])
                inX    = data['msi']   
                inX  = np.expand_dims(inX, axis = 0)
                inX = tf.to_float(inX);
                inX = sess.run(inX)
                psnr = skimage.measure.compare_psnr(inX, pred_X)
                print(files[i] + ' done!'+' psnr=%.4f', psnr)
                
                
if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')
    
    with tf.device(dev):
        if FLAGS.mode == 'test': # simple test
            test(0)
        elif FLAGS.mode == 'iniTest': # only for debug
            test(1)
        elif FLAGS.mode == 'testAll': # test all
            testAll()
        elif FLAGS.mode == 'train': # train
            train()
        elif FLAGS.mode == 'trainGAN': # trainGAN
            trainGAN()
        elif FLAGS.mode == 'testAllGAN': # trainGAN
            testAllGAN()


    
  
