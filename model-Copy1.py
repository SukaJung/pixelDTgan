import tensorflow as tf
import tensorflow.contrib.layers as tcl

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class Discriminator(object):

    def __init__(self):
        self.name = 'Discriminator'
    def __call__(self, z,ndf=96,reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            
            paddings = tf.constant([[0,0],[1,0], [1,0],[0,0]])
            z = tf.pad(z,paddings)
            Layer1 = tcl.conv2d(inputs=z, num_outputs=ndf, activation_fn=lrelu, stride=2, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID")
            Layer1= tf.pad(Layer1,paddings)
            Layer2 = tcl.conv2d(inputs=Layer1, num_outputs=ndf*2, activation_fn=lrelu, stride=2, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID",normalizer_fn=tcl.batch_norm)
            Layer2= tf.pad(Layer2,paddings)
            
            Layer3 = tcl.conv2d(inputs=Layer2, num_outputs=ndf*4, activation_fn=lrelu, stride=2, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID",normalizer_fn=tcl.batch_norm)
            Layer3= tf.pad(Layer3,paddings)
            Layer4 = tcl.conv2d(inputs=Layer3, num_outputs=ndf*8, activation_fn=lrelu, stride=2, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID",normalizer_fn=tcl.batch_norm)

            Layer5 = tcl.conv2d(inputs=Layer4, num_outputs=1, activation_fn=tf.nn.sigmoid, stride=4, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="SAME")
            return Layer5
        
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class DiscriminatorA(object):

    def __init__(self):
        self.name = 'DiscriminatorA'

    def __call__(self, z,ndf=96,reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            paddings = tf.constant([[0,0],[1,0], [1,0],[0,0]])
            z = tf.pad(z,paddings)
            Layer1 = tcl.conv2d(inputs=z, num_outputs=ndf, activation_fn=lrelu, stride=2, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID")
            Layer1= tf.pad(Layer1,paddings)
            Layer2 = tcl.conv2d(inputs=Layer1, num_outputs=ndf*2, activation_fn=lrelu, stride=2, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID",normalizer_fn=tcl.batch_norm)
            Layer2= tf.pad(Layer2,paddings)
            
            Layer3 = tcl.conv2d(inputs=Layer2, num_outputs=ndf*4, activation_fn=lrelu, stride=2, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID",normalizer_fn=tcl.batch_norm)
            Layer3= tf.pad(Layer3,paddings)
            Layer4 = tcl.conv2d(inputs=Layer3, num_outputs=ndf*8, activation_fn=lrelu, stride=2, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID",normalizer_fn=tcl.batch_norm)

            Layer5 = tcl.conv2d(inputs=Layer4, num_outputs=1, activation_fn=tf.nn.sigmoid, stride=4, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="SAME")
            return Layer5

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    

class Converter(object):

    def __init__(self):
        self.name = 'Converter'

    def __call__(self, z, nc=3, ngf=96,reuse=False):
        with tf.variable_scope(self.name) as scope:
            paddings = tf.constant([[0,0],[1,0], [1,0],[0,0]])
            z = tf.pad(z,paddings)
            Layer1 = tcl.conv2d(inputs=z,num_outputs=ngf,activation_fn=lrelu,stride=2,kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID")
            
            Layer1 = tf.pad(Layer1,paddings)
            Layer2 = tcl.conv2d(inputs=Layer1, num_outputs=ngf*2, activation_fn=lrelu, stride=2, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID",normalizer_fn=tcl.batch_norm)
            
            Layer2 = tf.pad(Layer2,paddings)
            Layer3 = tcl.conv2d(inputs=Layer2, num_outputs=ngf*4, activation_fn=lrelu, stride=2, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID",normalizer_fn=tcl.batch_norm)

            Layer3 = tf.pad(Layer3,paddings)
            Layer4 = tcl.conv2d(inputs=Layer3, num_outputs=ngf*8, activation_fn=lrelu, stride=2, kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID",normalizer_fn=tcl.batch_norm)

            Layer5 = tcl.conv2d_transpose(inputs=Layer4,num_outputs=ngf*4,activation_fn=tf.nn.relu,stride=2,kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID",normalizer_fn=tcl.batch_norm)

            Layer6 = tcl.conv2d_transpose(inputs=Layer5,num_outputs=ngf*2,activation_fn=tf.nn.relu,stride=2,kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID",normalizer_fn=tcl.batch_norm)
            Layer6 = tf.slice(Layer6,[0,1,1,0],[-1,16,16,-1])
            
            Layer7 = tcl.conv2d_transpose(inputs=Layer6,num_outputs=ngf,activation_fn=tf.nn.relu,stride=2,kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID",normalizer_fn=tcl.batch_norm)
            Layer7 = tf.slice(Layer7,[0,1,1,0],[-1,32,32,-1])

            
            Layer8 = tcl.conv2d_transpose(inputs=Layer7,num_outputs=nc,activation_fn=tf.nn.tanh,stride=2,kernel_size=4, weights_initializer=tf.random_normal_initializer(0, 0.02),padding="VALID")
            Layer8 = tf.slice(Layer8,[0,1,1,0],[-1,64,64,-1])

            return Layer8

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)