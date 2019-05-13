import os
import tensorflow as tf
import numpy as np
import cv2


def test_model(img_dir, model_path):
    # 获取图片总数
    input_count = 0
    for i in range(0, 10):
        dir = img_dir + '%s' % i  # 这里可以改成你自己的图片目录，i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                input_count += 1
    print(input_count)
    # 定义对应维数和各维长度的数组
    input_images = np.array([[[0, 0, 0]] * 48 * 32 for i in range(input_count)])
    input_labels = np.array([[0] * 10 for i in range(input_count)])

    # 读取图片和标签
    index = 0
    for i in range(10):
        dir = os.path.join(img_dir, str(i))  # i为分类标签
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                filename = os.path.join(rt, filename)
                # print(filename)
                img = cv2.imread(filename)
                height, width = 48, 32
                img = cv2.resize(img, (height, width))
                for h in range(0, height):
                    for w in range(0, width):
                        input_images[index][w + h * width] = img[w, h]
                print(index)
                input_labels[index][i] = 1
                index += 1

    with tf.Session() as sess:
        # 加载模型
        saver = tf.train.import_meta_graph('./models/models.meta')
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        # 从模型中加载变量
        x_ = graph.get_tensor_by_name("input_x:0")
        keep_prob = graph.get_tensor_by_name("Placeholder:0")
        y_ = graph.get_tensor_by_name("add_3:0")
        labels = graph.get_tensor_by_name("test_y:0")
        result = tf.argmax(tf.nn.softmax(y_), 1)  # 预测卡号
        accuracy = graph.get_tensor_by_name("Mean_1:0")  # 正确率计算

        print(sess.run(result, feed_dict={x_: input_images, labels: input_labels, keep_prob: 1.0}))  # 预测结果
        print(accuracy.eval(feed_dict={x_: input_images, labels: input_labels, keep_prob: 1.0}))  # 输出正确率


def main():
    img_dir = './test_img/'
    '''
    此处为测试图片目录
    格式为
    --test_img
        --0/
        --1/
        --2/
        ...
        ...
        --9/
    '''
    model_path = './models/'  # 此处为模型加载路径
    test_model(img_dir, model_path)


if __name__ == "__main__":
    main()
