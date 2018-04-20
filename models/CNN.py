import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.python.layers.base import Layer
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

class CNN_block(Layer):
    def __init__(self,out_kernel_size):
        self.cnn1=tf.layers.Conv2D(out_kernel_size,kernel_size=3,strides=(1, 1),padding='same',name='conv1')
        self.cnn2=tf.layers.Conv2D(out_kernel_size,kernel_size=3,strides=(1, 1),padding='same',name='conv2')
        self.cnn3=tf.layers.Conv2D(out_kernel_size,kernel_size=3,strides=(1, 1),padding='same',name='conv3')
        self.batch_norm1=tf.layers.BatchNormalization(axis=3,name='bn1')
        self.batch_norm2=tf.layers.BatchNormalization(axis=3,name='bn2')
        self.batch_norm3=tf.layers.BatchNormalization(axis=3,name='bn3')
        self.maxpool=tf.layers.MaxPooling2D([2,2],[2,2])
    
    def __call__(self,inputs,is_training):
        out1=tf.nn.relu(self.batch_norm1(inputs=self.cnn1(inputs),training=is_training))
        out2=tf.nn.relu(self.batch_norm2(inputs=self.cnn2(out1),training=is_training))
        out3=tf.nn.relu(self.batch_norm3(inputs=self.cnn3(out2),training=is_training))
        return self.maxpool(out3)
    
class CNN(Layer):
    def __init__(self):
        self.X=tf.placeholder(dtype=tf.float32,shape=[None,28,28,1],name='X')
        self.y=tf.placeholder(dtype=tf.int64,shape=[None,10],name='y')
        self.is_training=tf.placeholder(dtype=tf.bool,name='is_training')
        self.cnn_block1=CNN_block(64)
        self.cnn_block2=CNN_block(128)
        self.dense1=tf.layers.Dense(512)
        self.dense2=tf.layers.Dense(512)
        self.dense3=tf.layers.Dense(10)
        
    def __call__(self):
        out1=self.cnn_block1(self.X,self.is_training)
        out2=self.cnn_block2(out1,self.is_training)
        flt=tf.layers.flatten(out2)
        out1=tf.nn.relu(self.dense1(flt))
        out2=tf.nn.relu(self.dense2(out1))
        score=self.dense3(out2)
        loss=tf.losses.softmax_cross_entropy(onehot_labels=self.y,logits=score)
        predictions = tf.argmax(score, 1)
        correct_predictions = tf.equal(predictions, tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
        return loss,accuracy


if __name__=="__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("../mnist/", one_hot=True)
    num_train=mnist.train.num_examples
    num_val=mnist.validation.num_examples
    num_test=mnist.test.num_examples
    
    cnn=CNN()
    loss,accuracy=cnn()
    
    optimizier=tf.train.AdamOptimizer(learning_rate=1e-5)
    train_step = optimizier.minimize(loss)
    
    max_epoch=10
    batch_size=8
    print_every=625

    def train():
        num_iteration=num_train//batch_size
        for it in range(num_iteration):
            images,labels=mnist.train.next_batch(batch_size)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # update runing mean in batch normalization
            loss_num,_,__ = sess.run([loss,train_step,extra_update_ops],feed_dict={cnn.X:images.reshape(-1,28,28,1),cnn.y:labels,cnn.is_training:True})
            if it==0 or (it+1)%print_every==0 or it==num_iteration-1:
                print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                      'iteration %d/%d:' % (it+1,num_iteration),'current training loss = %f' % (loss_num))

    def eval(dataset,num_iteration):
        total_loss=0
        total_accuracy=0
        for it in range(num_iteration):
            images,labels=dataset.next_batch(batch_size)
            loss_num,accuracy_num = sess.run([loss,accuracy],feed_dict={cnn.X:images.reshape(-1,28,28,1), cnn.y:labels,cnn.is_training:False})
            total_loss+=loss_num
            total_accuracy+=accuracy_num
        total_loss/=num_iteration
        total_accuracy/=num_iteration
        return total_loss,total_accuracy
    
    
    acc_train_his=[]
    acc_val_his=[]
    acc_test_his=[]

    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        max_acc=None
        for epoch in range(max_epoch):
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'start epoch %d/%d:' % (epoch+1,max_epoch))
            train()
            loss_train,acc_train=eval(mnist.train,2000)
            loss_val,acc_val=eval(mnist.validation,500)
            loss_test,acc_test=eval(mnist.test,1000)
            acc_train_his.append(acc_train)
            acc_val_his.append(acc_val)
            acc_test_his.append(acc_test)

            if max_acc==None or acc_val>max_acc:
                max_acc=acc_val
                save_path = saver.save(sess, "parameters/CNN/CNN.ckpt")
                print("Currently maximum accuracy on validation set, model saved in path: %s" % save_path)

            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'end epoch %d/%d:' % (epoch+1,max_epoch),
                 'acc_train=%.3f%% acc_val=%.3f%% acc_test=%.3f%%' % (acc_train*100.0,acc_val*100.0,acc_test*100.0))
            
    plt.figure(1)
    ptr,=plt.plot(range(max_epoch),acc_train_his,'r-')
    pva,=plt.plot(range(max_epoch),acc_val_his,'b-')
    pte,=plt.plot(range(max_epoch),acc_test_his,'y-')
    plt.xlabel('training epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy on three sets')
    plt.legend((ptr,pva,pte),('train','validation','test'))
    plt.savefig('model-CNN.png')
    plt.show()
