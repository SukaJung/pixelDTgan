from model2 import *
from dataset import *
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot(samples,assimg,img):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(3, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    for i, sample in enumerate(assimg):
        ax = plt.subplot(gs[10+i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    for i, sample in enumerate(img):
        ax = plt.subplot(gs[20+i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    return fig

def testplot(samples1,samples2):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, sample in enumerate(samples1):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    for i, sample in enumerate(samples2):
        ax = plt.subplot(gs[len(samples1)+i])
        plt.axis('off')
        sample = (sample) * 255
        sample = sample.astype(np.uint8)
        plt.imshow(sample)
    return fig

def scaling_img(img):
    img -= np.mean(img)
    img /= np.std(img)
    min_ = np.min(img)
    max_ = np.max(img)
    img -= min_
    img /= (max_-min_)
    img=2*img - 1
    return img

def load_image(name):
    inImage_ = cv2.imread("./test/"+name)
    inImage = cv2.cvtColor(inImage_, cv2.COLOR_RGB2BGR)
    info = np.iinfo(inImage.dtype)
    inImage = inImage.astype(np.float) / info.max

    iw = inImage.shape[1]
    ih = inImage.shape[0]
    if iw < ih:
        inImage = cv2.resize(inImage, (64, int(64 * ih/iw)))
    else:
        inImage = cv2.resize(inImage, (int(64 * iw / ih), 64))
    inImage = inImage[0:64, 0:64]
    inImage = inImage[0:64, 0:64]
    return inImage


def read_testset():
    batch = []
    for i in range(1,4):
        batch.append(load_image("test{}.jpg".format(i)))
    return batch

class PixelDTgan():
    def __init__(self,converter,discriminator,discriminatorA,data):
        self.converter = converter
        self.discriminator = discriminator
        self.discriminatorA = discriminatorA

        self.data = data
           
        # data
        self.size = self.data.size
        self.channel = self.data.channel
        
        #learning rate
        self.lr = tf.placeholder(tf.float32,shape=[])
        
        #input data
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

        # loss
        self.D_loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_ass, labels=tf.ones_like(self.D_ass)))+ tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_noass, labels=tf.ones_like(self.D_noass)))+tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake))))/3
                                     
        self.A_loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.A_ass, labels=tf.ones_like(self.A_ass)))+ tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.A_noass, labels=tf.zeros_like(self.A_noass)))+tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.A_fake, labels=tf.zeros_like(self.A_fake))))/3

        self.C_loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))+tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.A_fake, labels=tf.ones_like(self.A_fake))))/2
        
        
        # solver
        self.D_optimizer = tf.train.AdamOptimizer(beta1=0.5,learning_rate=self.lr).minimize(self.D_loss, var_list=self.discriminator.vars)
        self.A_optimizer = tf.train.AdamOptimizer(beta1=0.5,learning_rate=self.lr).minimize(self.A_loss, var_list=self.discriminatorA.vars)
        self.C_optimizer = tf.train.AdamOptimizer(beta1=0.5,learning_rate=self.lr).minimize(self.C_loss, var_list=self.converter.vars)

        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self,training_epoch=1000000,batch_size=128):
        self.sess.run(tf.global_variables_initializer())
        path = "./model1/model1-4440.meta"
        self.loader = tf.train.import_meta_graph('./model1/model1-4440.meta')
        self.loader.restore(self.sess,tf.train.latest_checkpoint('./model1'))
        start_point = int(path.split('/')[-1].split('-')[-1].split(".")[0])
        print(start_point)
        for epoch in range(start_point,training_epoch):
            
            ass_label, noass_label, img = self.data.getbatch(batch_size)

#             ass_label = scaling_img(np.array(ass_label))
#             noass_label = scaling_img(np.array(noass_label))
            img = scaling_img(np.array(img))

            D_loss_curr, _ = self.sess.run([self.D_loss,self.D_optimizer],feed_dict={self.X: img, self.Y : ass_label,self.un_Y:noass_label,self.lr:0.0002/3})
            A_loss_curr, _ = self.sess.run([self.A_loss,self.A_optimizer],feed_dict={self.X: img, self.Y : ass_label,self.un_Y:noass_label,self.lr:0.0002/3})
            C_loss_curr, _ = self.sess.run([self.C_loss,self.C_optimizer],feed_dict={self.X: img, self.Y : ass_label,self.lr:0.0002/2})
            
            
            print('Iter: {}; C loss: {:.4},D loss: {:.4},A loss: {:.4} total loss : {}'.format(epoch, C_loss_curr, D_loss_curr, A_loss_curr,C_loss_curr+D_loss_curr+A_loss_curr))
            if epoch%30 == 0:
                test_set = scaling_img(read_testset())
                
                test_output = self.sess.run(self.G,feed_dict={self.X:test_set})
                fig = testplot(test_set,test_output)
                plt.savefig('outputs/test/test{}.png'.format(epoch), bbox_inches='tight')
                plt.close(fig)
                
                
                outputs = self.sess.run(self.G,feed_dict={self.X:img})
                fig = plot(outputs[0:10],img[0:10],ass_label[0:10])
                plt.savefig('outputs/{}.png'.format(epoch), bbox_inches='tight')
                plt.close(fig)
                 
            if epoch%20== 0:
                self.saver.save(self.sess, './model1/model1', global_step=epoch)
                
    def eval(checkpoint):
        self.sess.run(tf.global_variables_initializer())
        path = "./model1/model1-{}.meta".format(checkpoint)
        self.loader = tf.train.import_meta_graph('./model1/model1-{}.meta'.format(checkpoint))
        self.loader.restore(self.sess,tf.train.latest_checkpoint('./model1'))
        start_point = int(path.split('/')[-1].split('-')[-1].split(".")[0])
        test_set = scaling_img(read_testset())
                
        test_output = self.sess.run(self.G,feed_dict={self.X:test_set})
        fig = testplot(test_set,test_output)
        plt.savefig('outputs/test/test{}.png'.format(epoch), bbox_inches='tight')
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