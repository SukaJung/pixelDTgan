from model import *
from dataset import *
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot(samples,assimg,img):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(6, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        sample = (sample+1)/2 * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    for i, sample in enumerate(assimg):
        ax = plt.subplot(gs[20+i])
        plt.axis('off')
        sample = (sample+1)/2 * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    for i, sample in enumerate(img):
        ax = plt.subplot(gs[40+i])
        plt.axis('off')
        sample = (sample+1)/2 * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    return fig

def scaling_img(img):
    return img*2-1

class PixelDTgan():
    def __init__(self,converter,discriminator,discriminatorA,data):
        self.converter = converter
        self.discriminator = discriminator
        self.discriminatorA = discriminatorA
        
        self.data = data
           
        # data
        self.size = self.data.size
        self.channel = self.data.channel
        
        self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.un_Y = tf.placeholder(tf.float32,shape=[None,self.size,self.size,self.channel])
        self.Y = tf.placeholder(tf.float32,shape=[None,self.size,self.size,self.channel])
        
        # nets
        self.G = self.converter(self.X)
        
        self.D_ass   = self.discriminator(self.Y)
        self.D_noass = self.discriminator(self.un_Y,reuse=True)
        self.D_fake  = self.discriminator(self.G,reuse=True)
        
        self.A_ass   = self.discriminatorA(tf.concat([self.X,self.Y],3))
        self.A_noass = self.discriminatorA(tf.concat([self.X,self.un_Y],3),reuse=True)
        self.A_fake  = self.discriminatorA(tf.concat([self.X,self.G],3),reuse=True)
        
        self.C_ass = self.discriminator(self.G,reuse=True)
        self.C_noass = self.discriminatorA(tf.concat([self.X,self.G],3),reuse=True)
        
        # loss
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_ass, labels=tf.ones_like(self.D_ass)))+             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_noass, labels=tf.ones_like(self.D_noass)))+             tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
                                     
        self.A_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.A_ass, labels=tf.ones_like(self.A_ass)))+                   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.A_noass, labels=tf.zeros_like(self.A_noass)))+                   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.A_fake, labels=tf.zeros_like(self.A_fake)))
       
    
        self.C_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.C_ass, labels=tf.ones_like(self.C_ass)))+tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.C_noass, labels=tf.ones_like(self.C_noass)))
        
        
        # solver
        self.D_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002/3).minimize(self.D_loss,                                                   var_list=self.discriminator.vars)
        self.A_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002/3).minimize(self.A_loss,                                                   var_list=self.discriminatorA.vars)
        self.C_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002/2).minimize(self.C_loss,                                                   var_list=self.converter.vars)

        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self,training_epoch=1000000,batch_size=128):
        self.sess.run(tf.global_variables_initializer())
        num_img = 0
        for epoch in range(training_epoch):
            
            ass_label, noass_label, img = self.data.getbatch(batch_size)
            
            ass_label = scaling_img(np.array(ass_label))
            noass_label = scaling_img(np.array(noass_label))
            img = scaling_img(np.array(img))
            
            D_loss_curr, _ = self.sess.run([self.D_loss,self.D_optimizer],feed_dict={self.X: img, self.Y : ass_label,self.un_Y:noass_label})

            A_loss_curr, _ = self.sess.run([self.A_loss,self.A_optimizer],feed_dict={self.X: img, self.Y : ass_label,self.un_Y:noass_label})
     
            C_loss_curr, _ = self.sess.run([self.C_loss,self.C_optimizer],feed_dict={self.X: img, self.Y : ass_label})
            
           
            print('Iter: {}; C loss: {:.4},D loss: {:.4},A loss: {:.4} total loss : {}'.format(epoch, C_loss_curr/2, D_loss_curr/3, A_loss_curr/3,C_loss_curr/2+D_loss_curr/3+ A_loss_curr/3))
            if epoch%50 == 0:
                outputs = self.sess.run(self.G,feed_dict={self.X:img})
                fig = plot(outputs[0:20],img[0:20],ass_label[0:20])
                plt.savefig('outputs/{}.png'.format(epoch), bbox_inches='tight')
                plt.close(fig)
             



def main():
    converter = Converter()
    discriminator = Discriminator()
    discriminatora = DiscriminatorA()
    data = LookbookDataset(data_dir="/home/suka/eliceproject_dataset/lookbook/data/",index_dir="/home/suka/PycharmProjects/pixelDTgan/")
    
    
    # run
    pixeldtgan = PixelDTgan(converter, discriminator,discriminatora,data)
    pixeldtgan.train()

if __name__=="__main__":
    main()