# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 23:05:10 2018
这个版本里，让各层共享同一个下采样，且各通道也是用同一个下采样的，但是上采样不共享，这是目前最优的设置
@author: XieQi
"""
import tensorflow as tf
slim = tf.contrib.slim

def leaky_relu(x):
    return tf.where(tf.greater(x, 0), x, 0.01 * x)
# main HSISR net
def HSISRnet(Z, iniUp3x3,outDim,HSInetL,subnetL,ratio=2):
    


    CListX = getCs('TheCsX', ratio)#初始化X的下采样需求的核
    
    # 第一层
    G  = -UpSam('Z1',Z,iniUp3x3, outDim, ratio) # 残差上采样
    X = -G # YB残差
#    HY  = resCNNnet_addF(('Pri%s'%(1)),HY,Y,1,upRank,subnetL)
    X  = resCNNnet(('Pri%s'%(1)),X,1, outDim,subnetL)
    ListX = []
    
    # 第二到倒二层
    for j in range(HSInetL-2):
        ListX.append(X)
        downX2  = downSam( ('CX%s'%(j+2)),ListX[int(j)],CListX,outDim,  ratio)
        E   = downX2-Z
        G   = UpSam( ('E%s'%(j+2)),E,iniUp3x3, outDim, ratio)
        X  = X-G
#        HY  = resCNNnet_addF(('Pri%s'%(j+2)),HY,Y/10,j+2,upRank,subnetL)
        X  = resCNNnet(('Pri%s'%(j+2)),X,j+2, outDim,subnetL)
    
    #最后一层
    ListX.append(X)
    outX    = resCNNnet('FinalAjust',ListX[int(HSInetL-2)],101,outDim, levelN = 5)
    CX  = downSam( ('CX%s'%(HSInetL)),ListX[int(HSInetL-2)],CListX, outDim,  ratio)
    E  = CX-Z
    return outX, ListX,E



def HSISRCRNNnet(Z, iniUp3x3,outDim,HSInetL,subnetL,ratio=2):
    CListX = getCs('TheCsX', ratio)#初始化X的下采样需求的核
    ######################################参数初始化He /MSRA initialization
    ######定义feed-forward convolution的参数
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,  mode='FAN_AVG', 
                                                             uniform=False)
    W1_f = tf.Variable(initializer([3,3,outDim,64]))
    W2_f = tf.Variable(initializer([3,3,64,64]))
    W3_f = tf.Variable(initializer([3,3,64,64]))
    W4_f = tf.Variable(initializer([3,3,64,64]))
    W5_f = tf.Variable(initializer([3,3,64,outDim]))
    b1_f = tf.Variable(tf.zeros([64]))
    b2_f = tf.Variable(tf.zeros([64]))
    b3_f = tf.Variable(tf.zeros([64]))
    b4_f = tf.Variable(tf.zeros([64]))
    b5_f = tf.Variable(tf.zeros([outDim]))
    ######定义recurrent convolution(iteration)的参数
    W1_i = tf.Variable(initializer([3,3,64,64]))
    W2_i = tf.Variable(initializer([3,3,64,64]))
    W3_i = tf.Variable(initializer([3,3,64,64]))
    W4_i = tf.Variable(initializer([3,3,64,64]))
    
    H1 = []
    H2 = []
    H3 = []
    H4 = []
    H5 = []
    
    # 第一层
    G  = -UpSam('Z1',Z,iniUp3x3, outDim, ratio) # 残差上采样
    X = -G # YB残差

    H1.append(tf.nn.relu(tf.nn.conv2d(X, W1_f, strides = [1,1,1,1], padding = 'SAME') + b1_f))
    H2.append(tf.nn.relu(tf.nn.conv2d(H1[-1], W2_f, strides = [1,1,1,1], padding = 'SAME') + b2_f))
    H3.append(tf.nn.relu(tf.nn.conv2d(H2[-1], W3_f, strides = [1,1,1,1], padding = 'SAME') + b3_f))
    H4.append(tf.nn.relu(tf.nn.conv2d(H3[-1], W4_f, strides = [1,1,1,1], padding = 'SAME') + b4_f))
    H5.append(tf.nn.relu(tf.nn.conv2d(H4[-1], W5_f, strides = [1,1,1,1], padding = 'SAME') + b5_f))
    X = H5[-1]
    ListX = [] 

    # 第二到倒二层
    for i in range(HSInetL-2):
        ListX.append(X)
        downX2  = downSam( ('CX%s'%(i+2)),ListX[int(i)],CListX,outDim,  ratio)
        E   = downX2-Z
        G   = UpSam( ('E%s'%(i+2)),E,iniUp3x3, outDim, ratio)
        X  = X-G

        H1.append(tf.nn.relu(tf.nn.conv2d(X, W1_f, strides = [1,1,1,1], padding = 'SAME') 
                        + tf.nn.conv2d(H1[-1], W1_i, strides = [1,1,1,1], padding = 'SAME') + b1_f))
        H2.append(tf.nn.relu(tf.nn.conv2d(H1[-1], W2_f, strides = [1,1,1,1], padding = 'SAME') 
                        + tf.nn.conv2d(H2[-1], W2_i, strides = [1,1,1,1], padding = 'SAME') + b2_f))
        H3.append(tf.nn.relu(tf.nn.conv2d(H2[-1], W3_f, strides = [1,1,1,1], padding = 'SAME') 
                        + tf.nn.conv2d(H3[-1], W3_i, strides = [1,1,1,1], padding = 'SAME') + b3_f))
        H4.append(tf.nn.relu(tf.nn.conv2d(H3[-1], W4_f, strides = [1,1,1,1], padding = 'SAME') 
                        + tf.nn.conv2d(H4[-1], W4_i, strides = [1,1,1,1], padding = 'SAME') + b4_f))
        H5.append(tf.nn.relu(tf.nn.conv2d(H4[-1], W5_f, strides = [1,1,1,1], padding = 'SAME') + b5_f)) 
        X = H5[-1]
        
    
    #最后一层
    ListX.append(X)
    outX    = resCNNnet('FinalAjust',ListX[int(HSInetL-2)],101,outDim, levelN = 5)
    CX  = downSam( ('CX%s'%(HSInetL)),ListX[int(HSInetL-2)],CListX, outDim,  ratio)
    E  = CX-Z
    return outX, ListX,E

def Discriminator(dbatch, name = 'Discriminator'):
    with tf.variable_scope(name):
        net = slim.conv2d(dbatch, 64, 1, activation_fn = leaky_relu)
        
        ochannels = [64,128,128,256,256,512,512]
        stride = [2,1]
        for i in range(7):
            net = slim.conv2d(net, ochannels[i], 3, stride = stride[i%2], activation_fn = leaky_relu, normalizer_fn = slim.batch_norm, scope = 'block'+str(i))
        dense1 = slim.fully_connected(net, 1024, activation_fn = leaky_relu)
        dense2 = slim.fully_connected(dense1, 1, activation_fn = tf.nn.sigmoid)
        
        return dense2

def Discriminator_spectral(dbatch, name = 'Discriminator_spectral'):
    with tf.variable_scope(name):
        net = slim.conv2d(dbatch, 64, 1, activation_fn = leaky_relu)
        
        ochannels = [64,128,256,512,256,128,64,1]
        for i in range(8):
            if i < 7:
                net = slim.conv2d(net, ochannels[i], 1, activation_fn = leaky_relu, normalizer_fn = slim.batch_norm, scope = 'block'+str(i))
            else:
                net = slim.conv2d(net, ochannels[i], 1, activation_fn = tf.nn.sigmoid, scope = 'block'+str(i))
        
        return net    
    




















     
# main HSI net
def HSInet(Y,Z, iniUp3x3,iniA,upRank,outDim,HSInetL,subnetL,ratio=32):
    

    B = tf.get_variable(
              'B', 
              [1, 1, upRank, outDim],
              tf.float32, 
              initializer=tf.truncated_normal_initializer(mean = 0.0, stddev = 0.1))
    tranB = tf.transpose(B,perm = [0,1,3,2])
    CListX = getCs('TheCsX', ratio)#初始化X的下采样需求的核
    downY4, downY16, _ = downSam('GetPrior',Y, CListX, 3,ratio)# getPrior for upsample
    
    # 第一层
    YA = MyconvA( 'YA1', Y, 3, outDim, [1,1,1,1], iniA) #算YA
    _, _, downX32 = downSam('CX1',YA, CListX, outDim, ratio)  # 空间下采样
    E  = downX32-Z   # Z上的残差
    G  = UpSam('E1',E, downY4, downY16, Y, iniUp3x3, outDim, ratio) # 残差上采样
    G  = tf.nn.conv2d(G, tranB, [1,1,1,1], padding='SAME')
    HY = -G # YB残差
#    HY  = resCNNnet_addF(('Pri%s'%(1)),HY,Y,1,upRank,subnetL)
    HY  = resCNNnet(('Pri%s'%(1)),HY,1,upRank, subnetL)
    ListX = []
    
    # 第二到倒二层
    for j in range(HSInetL-2):
        HYB= tf.nn.conv2d(HY, B, [1,1,1,1], padding='SAME')
        ListX.append(YA + HYB)
        _, _, downX32  = downSam( ('CX%s'%(j+2)),ListX[int(j)],CListX,outDim,  ratio)
        E   = downX32-Z
        G   = UpSam( ('E%s'%(j+2)),E, downY4, downY16, Y, iniUp3x3, outDim, ratio)
        G   = tf.nn.conv2d(G, tranB, [1,1,1,1], padding='SAME')
        HY  = HY-G
#        HY  = resCNNnet_addF(('Pri%s'%(j+2)),HY,Y/10,j+2,upRank,subnetL)
        HY  = resCNNnet(('Pri%s'%(j+2)),HY,j+2,upRank, subnetL)
    
    #最后一层
    HYB     = tf.nn.conv2d(HY, B, [1,1,1,1], padding='SAME')
    ListX.append(YA + HYB)
    outX    = resCNNnet('FinalAjust',ListX[int(HSInetL-2)],101,outDim, levelN = 5)
    _,_,CX  = downSam( ('CX%s'%(HSInetL)),ListX[int(HSInetL-2)],CListX, outDim,  ratio)
    E  = CX-Z
    return outX, ListX, YA, E, HY
    
    

# reCNNnet 不确定要用这个还是上面那个
def resCNNnet(name,X,j,channel,levelN):
    with tf.variable_scope(name): 
        for i in range(levelN-1):
            X = resLevel(('resCNN_%s_%s'%(j,i+1)), 3, X, channel)                        
        return X    
    
# reCNNnet 不确定要用这个还是上面那个
def resCNNnet_addF(name,X,Y,j,channel,levelN):
    with tf.variable_scope(name): 
        for i in range(levelN-1):
            X = resLevel_addF(('resCNN_%s_%s'%(j,i+1)), 3, X, Y, channel,3)                         
        return X     
                    
# 用pooling下采样  
def getCs(name, ratio):
    Clist = []
    with tf.variable_scope(name):
        filter3 = tf.get_variable(
        'Cfilter3', 
        [3, 3, 1, 1],
        tf.float32, 
        initializer=tf.constant_initializer(1/9))     
        Clist.append(filter3)
    return Clist    
    
def downSam(name, X, Clist, ChDim, ratio):
    k=-1
    with tf.variable_scope(name):
        k      = k+1
        X      = tf.nn.depthwise_conv2d(X, tf.tile(Clist[int(k)],[1,1,ChDim,1]), strides=[1,1,1,1],padding='SAME')
        downX2 = X[:,0:-1:2,0:-1:2,:]
        return downX2
          
  
def UpSam(name,X, iniUp3x3, outDim, ratio):
    with tf.variable_scope(name):               
        X = UpsumLevel2('Cfilter1',X,iniUp3x3, outDim)# 2 倍上采样               
        filter1 = tf.get_variable(
          'Blur', [4, 4, outDim, 1], tf.float32, initializer=tf.constant_initializer(1/16))
        X = tf.nn.depthwise_conv2d(X,filter1,strides=[1,1,1,1],padding='SAME')        
        return X  
        

def resLevel(name, Fsize,X, Channel):
    with tf.variable_scope(name):
        # 两层调整
        kernel = create_kernel(name='weights1', shape=[Fsize, Fsize, Channel, Channel+3])
        biases = tf.Variable(tf.constant(0.0, shape=[Channel+3], dtype=tf.float32), trainable=True, name='biases1')
        scale = tf.Variable(tf.ones([Channel+3])/20, trainable=True, name=('scale1'))
        beta = tf.Variable(tf.zeros([Channel+3]), trainable=True, name=('beta1'))

        conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)
        
        # 加到新的test里
        kernel = create_kernel(name='weights3', shape=[Fsize, Fsize, Channel+3, Channel+3])
        biases = tf.Variable(tf.constant(0.0, shape=[Channel+3], dtype=tf.float32), trainable=True, name='biases3')
        scale = tf.Variable(tf.ones([Channel+3])/20, trainable=True, name=('scale3'))
        beta = tf.Variable(tf.zeros([Channel+3]), trainable=True, name=('beta3'))

        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)
        
        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)
        
        # 加到新的test里
        
        
        kernel = create_kernel(name='weights2', shape=[Fsize, Fsize, Channel+3, Channel])
        biases = tf.Variable(tf.constant(0.0, shape=[Channel], dtype=tf.float32), trainable=True, name='biases2')
        scale = tf.Variable(tf.ones([Channel])/20, trainable=True, name=('scale2'))
        beta = tf.Variable(tf.zeros([Channel]), trainable=True, name=('beta2'))

        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)        


        X = tf.add(X, feature_relu)  #  shortcut  
        return X
    
def resLevel_addF(name, Fsize, X, Y,ChannelX,ChannelY):
    with tf.variable_scope(name):
        # 两层调整
        Channel = ChannelX+ChannelY
        kernel = create_kernel(name='weights1', shape=[Fsize, Fsize, Channel, Channel+3])
        biases = tf.Variable(tf.constant(0.0, shape=[Channel+3], dtype=tf.float32), trainable=True, name='biases1')
        scale = tf.Variable(tf.ones([Channel+3])/100, trainable=True, name=('scale1'))
        beta = tf.Variable(tf.zeros([Channel+3]), trainable=True, name=('beta1'))

        conv = tf.nn.conv2d(tf.concat([X,Y],3), kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)
        
        # 我又加了一层
        kernel = create_kernel(name='weights2', shape=[Fsize, Fsize, Channel+3, Channel+3])
        biases = tf.Variable(tf.constant(0.0, shape=[ Channel+3], dtype=tf.float32), trainable=True, name='biases2')
        scale = tf.Variable(tf.ones([ Channel+3])/100, trainable=True, name=('scale2'))
        beta = tf.Variable(tf.zeros([ Channel+3]), trainable=True, name=('beta2'))

        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
        feature_relu = tf.nn.relu(feature_normal)
        #
        
        kernel = create_kernel(name='weights3', shape=[Fsize, Fsize, Channel+3, ChannelX])
        biases = tf.Variable(tf.constant(0.0, shape=[ChannelX], dtype=tf.float32), trainable=True, name='biases3')
        scale = tf.Variable(tf.ones([ChannelX])/100, trainable=True, name=('scale3'))
        beta = tf.Variable(tf.zeros([ChannelX]), trainable=True, name=('beta3'))

        conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var  = tf.nn.moments(feature,[0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        feature_relu = tf.nn.relu(feature_normal)

        X = tf.add(X, feature_relu)  #  shortcut  
        return X    
    
def ConLevel(name, Fsize, X, inC, outC):
    with tf.variable_scope(name):
        kernel = create_kernel(name=('weights'), shape=[Fsize, Fsize, inC, outC])
        biases = tf.Variable(tf.constant(0.0, shape=[outC], dtype=tf.float32), trainable=True, name=('biases'))
        scale = tf.Variable(tf.ones([outC]), trainable=True, name=('scale'))
        beta = tf.Variable(tf.zeros([outC]), trainable=True, name=('beta'))

        conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
        X = tf.nn.bias_add(conv, biases)

        mean, var = tf.nn.moments(X,[0, 1, 2])
        X = tf.nn.batch_normalization(X,  mean, var, beta, scale, 1e-5)

        X = tf.nn.relu(X)
        return X
 
def UpsumLevel2(name,X,iniUp2x2, outDim):
     # 2 倍上采样        
    filter1 = tf.get_variable(
          name, 
          [3, 3, outDim, outDim],
          tf.float32, 
#          initializer=tf.constant_initializer(0.0))
          initializer=tf.constant_initializer(iniUp2x2/4))
    
    sizeX   = tf.shape(X)*[1,2,2,1]
    X = tf.nn.conv2d_transpose (X,filter1, sizeX, strides=[1,2,2,1], padding='SAME')
    return X

#def UpsumLevel4(name,X,iniUp4x4,outDim):
#     # 2 倍上采样        
#    filter1 = tf.get_variable(
#          name, 
#          [4, 4, outDim, outDim],
#          tf.float32, 
##          initializer=tf.constant_initializer(0.0))
#          initializer=tf.constant_initializer(iniUp4x4/16))
#    
#    sizeX   = tf.shape(X)*[1,4,4,1]
#    X = tf.nn.conv2d_transpose (X,filter1, sizeX, strides=[1,4,4,1], padding='SAME')
#    return X
    
    
# A 1X1卷积   
def MyconvA( name, x, in_filters, out_filters, strides, iniA):
    with tf.variable_scope(name):
      # 获取或新建卷积核，
      
#      iniA = [[ 0.05442263, -0.02493124, -0.0397463,  -0.08398886, -0.10482746, -0.1390029,
#              -0.15541755, -0.17359568, -0.19168225, -0.20186915, -0.22137468, -0.22697504,
#              -0.2082457,  -0.16593517, -0.12181907, -0.07575565, -0.02412732,  0.04199919,
#               0.13807541,  0.28079346,  0.44277778,  0.5692637,   0.63139963,  0.6594861,
#               0.6744813,   0.6769856,   0.67531425,  0.6628369,   0.6680167,  0.6780877,
#               0.68491614],
#             [ 0.04297409,  0.06621623,  0.07825829,  0.07871637,  0.09492727,  0.12790947,
#               0.16793568,  0.21970585,  0.29888564,  0.40316212,  0.5569614,   0.7663978,
#               0.88787705,  0.8999426,   0.872026,    0.8223045,   0.77352357,  0.7219894,
#               0.6331659,   0.45682576,  0.25250176,  0.09103899,  0.0128423,  -0.03040085,
#              -0.06139304, -0.06927183, -0.06893785, -0.06472709, -0.06605597, -0.07127436,
#              -0.08600657],
#             [ 0.29711956,  0.44763547,  0.46287882,  0.52228314,  0.540729,    0.5282523,
#               0.49282253,  0.46240586,  0.41122892,  0.30727905,  0.1805829,  -0.00943106,
#              -0.13869575, -0.19865918, -0.23038715, -0.24272782, -0.25109097, -0.25916272,
#              -0.25455403, -0.2154364,  -0.16579448, -0.1191813,  -0.0922951,  -0.0737847,
#              -0.05726132, -0.04480614, -0.03247489, -0.01654494, -0.00175067,  0.01205034,
#               0.04141013]]
      
      
      kernel = tf.get_variable(
              'A', 
              [1, 1, in_filters, out_filters],
              tf.float32, 
#              initializer=tf.constant_initializer(1))
              initializer=tf.constant_initializer(iniA))
      # 计算卷积
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')
  
    
# B 1X1卷积   
def MyconvB( name, x, in_filters, out_filters, strides):
    with tf.variable_scope(name):
      # 获取或新建卷积核，正态随机初始化
      kernel = tf.get_variable(
              'B', 
              [1, 1, in_filters, out_filters],
              tf.float32, 
              initializer=tf.truncated_normal_initializer(mean = 0.0, stddev = 0.1))
      # 计算卷积
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')    

def create_kernel(name, shape, initializer=tf.truncated_normal_initializer(mean = 0, stddev = 0.1)):
#def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
    return new_variables