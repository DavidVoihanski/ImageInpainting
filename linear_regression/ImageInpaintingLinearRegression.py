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


# this function saves the loss to csv file
def write_losses_to_csv(loss_list, loss_avg, s):
    loss_list.append(loss_avg)
    with open("loss_linear_regression_{}.csv".format(s), "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(loss_list)


def train():
    # the number of features, pixels above the area that we want to predict
    num_features = 780
    # alpha, steps size
    alpha = 0.00000001
    batch_size = 50
    # matrix in size - [size of batch][num of features]
    x = tf.placeholder(tf.float32, [None, num_features], name="x")
    # the real value of the pixel that we want to predict
    y_ = tf.placeholder(tf.float32, [None, 1], name="y_")
    # W, weights- [num of features][1]
    W = tf.Variable(tf.zeros([num_features, 1]), name="W")
    # b, bias - scalar
    b = tf.Variable(tf.zeros([1]), name="b")
    # our model function - y = x*W+b
    y = tf.add(tf.matmul(x, W), b)
    # loss function
    loss = tf.reduce_mean(tf.pow(y - y_, 2))
    update = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    # tensorbord var
    loss_sum = tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()

    sess = tf.Session()
    file_writer = tf.summary.FileWriter('./my_graph_linear_regression', sess.graph)
    sess.run(tf.global_variables_initializer())
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
                looss_eval = (loss.eval(session=sess, feed_dict={x: data_x, y_: data_y})) ** 0.5
                loss_pixel_list.append(looss_eval)
                if i % 100 == 0:
                    print('Iteration: ', i, ' pixel_i, pixel_j: ', (pixel_i, pixel_j), ' loss:', looss_eval)

                # update the pixel we just predicted
                pred = np.add(np.matmul(data_x, sess.run(W)), sess.run(b))
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

        # saves the model after each batch
        path_to_save = "/home/shaynaor/Desktop/shay-study/year_3/semester_1/deep_lerning/Image_Inpainting/linear_regression_2/linear_regression_model/image_inpainting_linear_regression"
        saver = tf.train.Saver()
        saver.save(sess, path_to_save)


    print("FINISH THE TRAIN")
    loss_avg = statistics.mean(loss_batch_list)
    print("Avg loss for all train images", loss_avg)
    write_losses_to_csv(loss_batch_list, loss_avg, "train")
    file_writer.close()


def test():
    # reads the test images
    images_path = "/home/shaynaor/Downloads/image_inpainting_dataset/test_images_batches"
    path_to_restore = "/home/shaynaor/Desktop/shay-study/year_3/semester_1/deep_lerning/Image_Inpainting/linear_regression_2/linear_regression_model/image_inpainting_linear_regression"



    sess = tf.Session()
    saver = tf.train.Saver()

    saver.restore(sess, path_to_restore)

    graph = tf.get_default_graph()
    W = graph.get_tensor_by_name("W:0")
    b = graph.get_tensor_by_name("b:0")
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    y = tf.add(tf.matmul(x, W), b)
    loss = tf.reduce_mean(tf.pow(y - y_, 2))


    loss_batch_list = []
    loss_pixel_list = []

    num_of_batches = 729
    counter_batch = 0
    batch_size = 50
    num_features = 780


    for i in range(0, num_of_batches):
        counter_batch = next_batch(counter_batch, images_path, num_of_batches)
        # ---------------
        # pic_counter = 0
        # for img in img_holes_list:
        #     cv2.imwrite("/home/shaynaor/Desktop/shay-study/year_3/semester_1/deep_lerning/Image_Inpainting/linear_regression_2/evaluetion/before/{}.jpeg".format(pic_counter), img)
        #     pic_counter += 1
        # ---------------
        for pixel_i in range(0, size_hole):
            for pixel_j in range(0, size_hole):
                data_x, data_y = find_data_x_y(batch_size, pixel_i, pixel_j, num_features)
                pred = np.add(np.matmul(data_x, sess.run(W)), sess.run(b))

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

        # ------------------
        # pic_counter = 0
        # for img in img_holes_list:
        #     cv2.imwrite("/home/shaynaor/Desktop/shay-study/year_3/semester_1/deep_lerning/Image_Inpainting/linear_regression_2/evaluetion/after/{}.jpeg".format(pic_counter), img)
        #     pic_counter += 1
        # ------------------



    print("FINISH THE TEST")
    loss_avg = statistics.mean(loss_batch_list)
    print("Avg loss for all test images", loss_avg)
    write_losses_to_csv(loss_batch_list, loss_avg, "test")


if __name__ == "__main__":
    train()
    test()
