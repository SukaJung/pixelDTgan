#hello_world.py

from flask import Flask
import tensorflow as tf
from train import *
from model import *
app = Flask(__name__)
gpu_options = tf.GPUOptions(allow_growth=True)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

loader = tf.train.import_meta_graph('./model/model-9100.meta')
loader.restore(sess, tf.train.latest_checkpoint('./model'))
sess.run(tf.global_variables_initializer())

X = tf.placeholder(tf.float32, shape=[None, 64,64, 3])

converter = Converter()
G = converter(X)

@app.route("/")
def eval():

    test_set = scaling_img(read_testset())
    print(np.max(test_set))


    test_output = sess.run(G, feed_dict={X: test_set})
    # fig = testplot(test_set, test_output)
    # fig.show()
    return "AA"
