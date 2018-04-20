import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
from tensorflow.python.layers.base import Layer
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

class Config(object):
    def __init__(self):
        self.batch_size=64
        self.img_size=28
        self.RNN_unit=256
        self.N_watch=10
        
config=Config()
        
class SRAM(Layer):
    def __init__(self):
        self.X=tf.placeholder(dtype=tf.float32,shape=[None,config.img_size*config.img_size],name='X')
        self.y=tf.placeholder(dtype=tf.int64,shape=[None,10],name='y')
        self.emission_net_low=tf.layers.Dense(units=512,name='low',_reuse=tf.AUTO_REUSE)
        self.emission_net_high=tf.layers.Dense(units=config.img_size*config.img_size,name='high',_reuse=tf.AUTO_REUSE)
        self.predict_net=tf.layers.Dense(units=10)
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(config.RNN_unit, state_is_tuple=True)
        self.mask_his=[]

    def get_next_input(self,output, i):
        emis=self.emission_net_high(tf.nn.relu(self.emission_net_low(output)))
        attention_weight=tf.nn.softmax(emis)
        self.mask_his.append(attention_weight)
        weighted_graph=self.X*attention_weight
        return weighted_graph
    
    def __call__(self):
        this_size=tf.shape(self.X)[0]
        init_state = self.lstm_cell.zero_state(this_size, tf.float32)
        inputs=[self.X]
        inputs.extend([0]*config.N_watch)
        self.mask_his=[]
        outputs,_ = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, init_state, self.lstm_cell, loop_function=self.get_next_input)
        output=outputs[-1]
        score=self.predict_net(output)
        predictions = tf.argmax(score, 1)
        correct_predictions = tf.equal(predictions, tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
        loss=tf.losses.softmax_cross_entropy(onehot_labels=self.y,logits=score)
        return loss,accuracy

    
if __name__=="__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("../mnist/", one_hot=True)
    num_train=mnist.train.num_examples
    num_val=mnist.validation.num_examples
    num_test=mnist.test.num_examples
    
    sram=SRAM()
    loss,accuracy=sram()
    
    optimizier=tf.train.AdamOptimizer(learning_rate=1e-5)
    train_step = optimizier.minimize(loss)
    
    max_epoch=100
    print_every=200

    def train():
        num_iteration=num_train//config.batch_size
        for it in range(num_iteration):
            images,labels=mnist.train.next_batch(config.batch_size)
            loss_num,_ = sess.run([loss,train_step],feed_dict={sram.X:images,sram.y:labels})
            if it==0 or (it+1)%print_every==0 or it==num_iteration-1:
                print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                      'iteration %d/%d:' % (it+1,num_iteration),'current training loss = %f' % (loss_num))

    def eval(dataset,num_iteration):
        total_loss=0
        total_accuracy=0
        for it in range(num_iteration):
            images,labels=dataset.next_batch(config.batch_size)
            loss_num,accuracy_num = sess.run([loss,accuracy],feed_dict={sram.X:images,sram.y:labels})
            total_loss+=loss_num
            total_accuracy+=accuracy_num
        total_loss/=num_iteration
        total_accuracy/=num_iteration
        return total_loss,total_accuracy
    
    acc_train_his=[]
    acc_val_his=[]
    acc_test_his=[]

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        max_acc=None
        for epoch in range(max_epoch):
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'start epoch %d/%d:' % (epoch+1,max_epoch))
            train()
            loss_train,acc_train=eval(mnist.train,500)
            loss_val,acc_val=eval(mnist.validation,70)
            loss_test,acc_test=eval(mnist.test,150)
            acc_train_his.append(acc_train)
            acc_val_his.append(acc_val)
            acc_test_his.append(acc_test)

            if max_acc==None or acc_val>max_acc:
                max_acc=acc_val
                save_path = saver.save(sess, "parameters/SRAM/SRAM.ckpt")
                print("Currently maximum accuracy on validation set, model saved in path: %s" % save_path)

            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'end epoch %d/%d:' % (epoch+1,max_epoch),
                 'acc_train=%.3f%% acc_val=%.3f%% acc_test=%.3f%%' % (acc_train*100.0,acc_val*100.0,acc_test*100.0))
            
    plt.figure(2)
    ptr,=plt.plot(range(max_epoch),acc_train_his,'r-')
    pva,=plt.plot(range(max_epoch),acc_val_his,'b-')
    pte,=plt.plot(range(max_epoch),acc_test_his,'y-')
    plt.xlabel('training epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy on three sets')
    plt.legend((ptr,pva,pte),('train','validation','test'))
    plt.savefig('model-SRAM.png')
    plt.show()
