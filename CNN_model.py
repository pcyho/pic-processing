import os
import tensorflow as tf
import numpy as np
import cv2

def CNN_train(img_dir):
    # 获取图片总数
    input_count = 0
    for rt, dirs, files in os.walk(img_dir):
        for filename in files:
            input_count += 1
    
    # 定义对应维数和各维长度的数组
    input_images = np.array([[[0,0,0]]*1536 for i in range(input_count)])
    input_labels = np.array([[0]*10 for i in range(input_count)])
    
    # 遍历图片目录生成图片数据和标签
    index = 0
    for i in range(0,10):
        dir = os.path.join(img_dir, str(i))  # i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = os.path.join(rt, filename)
                img = cv2.imread(filename)
                height, width = 48, 32
                img = cv2.resize(img, (height, width))
                for h in range(0, height):
                    for w in range(0, width):
                        input_images[index][w+h*width] = img[w,h]
                input_labels[index][i] = 1
                index += 1
    
    # 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
    x = tf.placeholder(tf.float32, shape=[None, 1536, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    x_image = tf.reshape(x, [-1, 48, 32, 3])
    
    # 定义第一个卷积层的variables和ops
    W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
    
    L1_conv = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    L1_relu = tf.nn.relu(L1_conv + b_conv1)
    L1_pool = tf.nn.max_pool(L1_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    #局部响应归一化
    norm1 = tf.nn.lrn(L1_pool, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # 定义第二个卷积层的variables和ops
    W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 64, 16], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[16]))
    
    L2_conv = tf.nn.conv2d(norm1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    L2_relu = tf.nn.relu(L2_conv + b_conv2)
    norm2 = tf.nn.lrn(L2_relu, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    L2_pool = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    # 全连接层
    W_fc1 = tf.Variable(tf.truncated_normal([12 * 8 * 16, 128], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[128]))
    
    h_pool2_flat = tf.reshape(L2_pool, [-1, 12*8*16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # readout层
    W_fc2 = tf.Variable(tf.truncated_normal([128, 10], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    # 定义优化器和训练op
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    ckpt_file_path = "./models"
    path = os.path.dirname(os.path.abspath(ckpt_file_path))
    if os.path.isdir(path) is False:
        os.makedirs(path)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print ("一共读取了 %s 个输入图像， %s 个标签" % (input_count, input_count))
    
        # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
        batch_size = 60
        iterations = 1000
        batches_count = int(input_count / batch_size)
        remainder = input_count % batch_size
        print ("数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count+1, batch_size, remainder))
    
        # 执行训练迭代
        for it in range(iterations):
            # 把输入数组转为np.array
            for n in range(batches_count):
                train_step.run(feed_dict={x: input_images[n*batch_size:(n+1)*batch_size], y_: input_labels[n*batch_size:(n+1)*batch_size], keep_prob: 0.5})
            if remainder > 0:
                start_index = batches_count * batch_size
                train_step.run(feed_dict={x: input_images[start_index:input_count-1], y_: input_labels[start_index:input_count-1], keep_prob: 0.5})
    
            # 每完成五次迭代，判断准确度
            iterate_accuracy = 0
            if it%5 == 0:
                iterate_accuracy = accuracy.eval(feed_dict={x: input_images, y_: input_labels, keep_prob: 1.0})
                print ('iteration %d: accuracy %s' % (it, iterate_accuracy))
                if iterate_accuracy >= 0.99:
                    tf.train.Saver().save(sess, ckpt_file_path, write_meta_graph=True)
                    break
    
        print ('完成训练!')

def main():
    img_dir = './more'
    CNN_train(img_dir)

if __name__ == '__main__':
    main()