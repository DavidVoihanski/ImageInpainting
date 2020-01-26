import cv2
import numpy as np
import tensorflow as tf
import statistics
from pathlib import Path
import csv

global img_list
img_list = []

global img_holes_list
img_holes_list = []

global hole_indexes
hole_indexes = []

global features_list
features_list = []

global y_true
y_true = []

global size_hole
size_hole = 10


# this function saves the loss to csv file
def write_losses_to_csv(loss_list, loss_avg, s):
    loss_list.append(loss_avg)
    with open("loss_MLP_{}.csv".format(s), "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(loss_list)


# this function, makes rectangle hole of sizeXsize starting from random i and random j
# and save the hole_image in img_holes_list
def make_hole_list():
    img_holes_list.clear()
    hole_indexes.clear()
    for img in img_list:
        start_i = np.random.random_integers(31, 215)
        start_j = np.random.random_integers(31, 215)
        hole_indexes.append((start_i, start_j))
        gray_img = img.copy()
        gray_img[start_j:start_j + size_hole, start_i: start_i + size_hole] = 0
        img_holes_list.append(gray_img)


# this function iterate on img_holes_list takes the starting i, j from hole_indexes
# and saves 10X10 matrix of pixels these matrix will be the features
def find_features(i, j):
    features_list.clear()
    counter = 0
    for hole_image in img_holes_list:
        start_i, start_j = hole_indexes[counter]
        start_i += i
        start_j += j
        counter += 1
        features_mat = hole_image[start_j - 19:start_j + 1, start_i - 19:start_i + 21]
        features_list_flat = np.array(features_mat).flat
        features_list_flat = features_list_flat[:len(features_list_flat) - 20]
        features_list.append(features_list_flat)


# this function receives path to directory converts all the images in the directory
# to gray scale and adds them to img_list
def create_image_array(dir_path):
    img_list.clear()
    path_list = Path(dir_path).glob("*")
    for path in path_list:
        img_path = str(path)
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_list.append(gray_img)


# this function returns the true value of the current batch pixels
def find_y_true(i, j):
    y_true.clear()
    counter = 0
    for img in img_list:
        start_i, start_j = hole_indexes[counter]
        start_i += i
        start_j += j
        counter += 1
        y_true.append(img[start_j, start_i])


# this function receives i,j and returns features list,y_true
def find_data_x_y(batch_size, pixel_i, pixel_j, num_features):
    find_features(pixel_i, pixel_j)
    data_x = np.array(features_list).reshape(-1, num_features)
    find_y_true(pixel_i, pixel_j)
    data_y = np.array(y_true).reshape(batch_size, 1)
    return data_x, data_y


def next_batch(counter_batch, path, num_dir):
    images_path = path + "/{}".format(counter_batch)
    print("Start to read the images from batch number: ", counter_batch)
    counter_batch += 1

    if counter_batch == num_dir:
        print("Starts new epoc")
        counter_batch = 0

    create_image_array(images_path)
    np.random.shuffle(img_list)
    # define the size of hole in the images
    make_hole_list()

    return counter_batch



def train():

    # the number of features, pixels above the area that we want to predict
    num_features = 780
    # alpha, steps size
    alpha = 0.000001
    batch_size = 50
    # three layers of hidden nodes the first has 100, the second 50 and the third 25 nodes.
    (hidden1_size, hidden2_size, hidden3_size) = (100, 50, 25)

    # x is a matrix [batch size][num of features]
    x = tf.placeholder(tf.float32, [None, num_features], name="x")
    # y_ the real value of the pixel that we want to predict
    y_ = tf.placeholder(tf.float32, [None, 1], name="y_")
    # W1 is a matrix [num of features][number of nodes in the first hidden layer], init with randomized values
    W1 = tf.Variable(tf.truncated_normal([num_features, hidden1_size], stddev=0.1), name="W1")
    # b is a vector of 0.1 in len of the number of nodes in the first hidden layer
    b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]), name="b1")
    # z1 = if x*W1+b > 0 then x*W1+b else 0
    z1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1), name="z1")
    # W2 is a matrix [number of nodes in the first hidden layer][number of nodes in the second hidden layer],
    # init with randomized values
    W2 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=0.1), name="W2")
    b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]), name="b2")
    z2 = tf.nn.relu(tf.add(tf.matmul(z1, W2), b2), name="z2")

    W3 = tf.Variable(tf.truncated_normal([hidden2_size, hidden3_size], stddev=0.1), name="W3")
    b3 = tf.Variable(tf.constant(0.1, shape=[hidden3_size]), name="b3")
    z3 = tf.nn.relu(tf.add(tf.matmul(z2, W3), b3), name="z3")

    W4 = tf.Variable(tf.truncated_normal([hidden3_size, 1], stddev=0.1), name="W4")
    b4 = tf.Variable(tf.constant(0.1, shape=[1]), name="b4")
    y = tf.add(tf.matmul(z3, W4), b4)

    # loss function
    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    update = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    # tensorbord var
    loss_sum = tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    file_writer = tf.summary.FileWriter('./my_graph_MLP1', sess.graph)
    sess.run(init)
    num_of_batches = 6569
    graph_counter = 0
    loss_batch_list = []
    loss_pixel_list = []
    counter_batch = 0
    path = "/home/shaynaor/Downloads/image_inpainting_dataset/train_images_batches"

    for i in range(0, 2*num_of_batches):
        counter_batch = next_batch(counter_batch, path, num_of_batches)
        for pixel_i in range(0, size_hole):
            for pixel_j in range(0, size_hole):

                data_x, data_y = find_data_x_y(batch_size, pixel_i, pixel_j, num_features)
                [_, curr_summary] = sess.run([update, merged], feed_dict={x: data_x, y_: data_y})
                file_writer.add_summary(curr_summary, graph_counter)
                graph_counter += 1
                loss_pixel_list.append((loss.eval(session=sess, feed_dict={x: data_x, y_: data_y})) ** 0.5)
                if i % 100 == 0:
                    print('Iteration: ', i, ' pixel_i, pixel_j: ', (pixel_i, pixel_j), ' loss:',
                          (loss.eval(session=sess, feed_dict={x: data_x, y_: data_y})) ** 0.5)



                # update the pixel we just predicted
                pred = np.add(np.matmul(sess.run(z3, feed_dict={x: data_x, y_: data_y}), sess.run(W4)), sess.run(b4))
                for k in range(0, batch_size):
                    start_i, start_j = hole_indexes[k]
                    start_i += pixel_i
                    start_j += pixel_j

                    pred_val = int(round(pred[k][0]))
                    img_holes_list[k][start_j, start_i] = pred_val

        batch_loss_avg = statistics.mean(loss_pixel_list)
        print("Loss AVG for batch number {} is {}".format(i, batch_loss_avg))
        loss_pixel_list.clear()
        loss_batch_list.append(batch_loss_avg)

        # saves the model after number of iteration_each_pixel
        path_to_save = "/home/shaynaor/Desktop/shay-study/year_3/semester_1/deep_lerning/image_inpainting/multilayer_perceptron/MLP_model/image_inpainting_MLP"
        saver = tf.train.Saver()
        saver.save(sess, path_to_save)
        file_writer.close()

    print("FINISH THE TRAIN")
    loss_avg = statistics.mean(loss_batch_list)
    print("Avg loss for all train images", loss_avg)
    write_losses_to_csv(loss_batch_list, loss_avg, "train")


def test():
    # reads the test images
    images_path = "/home/shaynaor/Downloads/image_inpainting_dataset/test_images_batches"

    num_of_batches = 729
    counter_batch = 0
    batch_size = 50
    num_features = 780

    sess = tf.Session()
    saver = tf.train.Saver()
    path_to_restore = "/home/shaynaor/Desktop/shay-study/year_3/semester_1/deep_lerning/Image_Inpainting/multilayer_perceptron/MLP_model/image_inpainting_MLP"
    saver.restore(sess, path_to_restore)

    graph = tf.get_default_graph()
    W4 = graph.get_tensor_by_name("W4:0")
    b4 = graph.get_tensor_by_name("b4:0")
    z3 = graph.get_tensor_by_name("z3:0")
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    y = tf.add(tf.matmul(z3, W4), b4)
    loss = tf.reduce_mean(tf.pow(y - y_, 2))

    loss_batch_list = []
    loss_pixel_list = []



    for i in range(0, num_of_batches):
        counter_batch = next_batch(counter_batch, images_path, num_of_batches)
        for pixel_i in range(0, size_hole):
            for pixel_j in range(0, size_hole):
                data_x, data_y = find_data_x_y(batch_size, pixel_i, pixel_j, num_features)
                pred = np.add(np.matmul(sess.run(z3, feed_dict={x: features_list, y_: data_y}), sess.run(W4)), sess.run(b4))
                loss_pixel_list.append((loss.eval(session=sess, feed_dict={x: data_x, y_: data_y})) ** 0.5)

                for k in range(0, batch_size):
                    start_i, start_j = hole_indexes[k]
                    start_i += pixel_i
                    start_j += pixel_j

                    pred_val = int(round(pred[k][0]))
                    img_holes_list[k][start_j, start_i] = pred_val

        batch_loss_avg = statistics.mean(loss_pixel_list)
        print("Loss AVG for batch number {} is {}".format(i, batch_loss_avg))
        loss_pixel_list.clear()
        loss_batch_list.append(batch_loss_avg)

    print("FINISH THE TEST")
    loss_avg = statistics.mean(loss_batch_list)
    print("Avg loss for all test images", loss_avg)
    write_losses_to_csv(loss_batch_list, loss_avg, "test")


if __name__ == "__main__":
    train()
    test()
