import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.python.layers.base import Layer
from tensorflow.contrib.legacy_seq2seq import rnn_decoder
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

class Config(object):
    def __init__(self):
        # parameters
        self.batch_size=64
        self.img_size=28
        self.sensor_unit=256
        self.lstm_size=256
        self.N_glimpse=7
        self.MC_test=128
        self.loc_std=0.5
        self.tot_size=self.batch_size*self.MC_test
        
config=Config()

class Glimpse_Network(Layer):
    def __init__(self):
        self.glimspe_size=[4,5,6]
        self.concat_size=5
        self.img_net=tf.layers.Dense(units=config.sensor_unit,name='glimpse_net/img_net')
        self.loc_net=tf.layers.Dense(units=config.sensor_unit,name='glimpse_net/loc_net')
        
    def glimpse_sensor(self,image,loc):
        glimpses_list=[tf.image.extract_glimpse(input=image,size=[gs,gs],offsets=loc) for gs in self.glimspe_size]
        glimpses_norm=[tf.image.resize_bilinear(g,[self.concat_size,self.concat_size]) for g in glimpses_list]
        glimpses=tf.concat(values=glimpses_norm,axis=3)  # batch_size*concat_size*concat_size*3
        return glimpses
    
    def __call__(self,image,loc):
        glimpses=self.glimpse_sensor(image,loc) # tot_size*concat_size*concat_size*3
        glimpses=tf.stop_gradient(glimpses)  # gradient has no need to flow through glimpses
        g_image=tf.nn.relu(self.img_net(inputs=tf.layers.flatten(glimpses)))
        g_loc=tf.nn.relu(self.loc_net(inputs=loc))
        g_out=tf.nn.relu(g_image+g_loc)
        return g_out
    
    
class RAM(Layer):
    def __init__(self):
        self.X=tf.placeholder(dtype=tf.float32,shape=[None,28,28,1])
        self.y=tf.placeholder(dtype=tf.int64,shape=[None,10])
        self.gNet=Glimpse_Network()
        self.lstm_cell = tf.contrib.rnn.LSTMCell(config.lstm_size)
        
        # emission_net_low=tf.layers.Dense(units=128,name='emission_net_low')
        self.emission_net_high=tf.layers.Dense(units=2,name='RAM/emission_net_high',_reuse=tf.AUTO_REUSE)
        self.baseline_net_low=tf.layers.Dense(units=128,name='RAM/baseline_net_low',_reuse=tf.AUTO_REUSE)
        self.baseline_net_mid=tf.layers.Dense(units=128,name='RAM/baseline_net_mid',_reuse=tf.AUTO_REUSE)
        self.baseline_net_high=tf.layers.Dense(units=1,name='RAM/baseline_net_high',_reuse=tf.AUTO_REUSE)
        self.predict_net=tf.layers.Dense(units=10,name='RAM/predict_net')
        
        self.loc_his=[]
        self.loglikelihood_his=[]
        self.baseline_his=[]
        
    def loglikelihood(self,sample,mean):
        gaussian=tf.distributions.Normal(loc=mean,scale=tf.constant([config.loc_std,config.loc_std]))
        llh=-gaussian.log_prob(sample)
        return tf.reduce_sum(llh,axis=1)
    
    def get_next_input(self,output, i):
        # emit mean of location
        loc_mean=self.emission_net_high(output)
        
        # sample next location by gaussian distribution centered at loc_mean
        loc_sample=tf.random_normal(shape=tf.shape(loc_mean),mean=loc_mean,stddev=config.loc_std)
        loc_sample=tf.stop_gradient(loc_sample)  # very important ***
        
        # calculate the -loglikelihood of the sampled position
        llh=self.loglikelihood(loc_sample,loc_mean)
        self.loglikelihood_his.append(llh)
        
        # normalize the location for next input
        normalized_loc=tf.tanh(loc_sample)
        self.loc_his.append(normalized_loc)
        normalized_loc=tf.stop_gradient(normalized_loc)

        # output deep accurate baseline(value) network
        baseline=self.baseline_net_high(tf.nn.relu(self.baseline_net_mid(tf.nn.relu(self.baseline_net_low(output)))))
        self.baseline_his.append(tf.squeeze(baseline))
        
        # prepare next input
        glimpses_out=self.gNet(self.X,normalized_loc)
        return glimpses_out
    
    def __call__(self):
        this_size=tf.shape(self.X)[0]
        start_location=tf.random_uniform(shape=[this_size,2],minval=-1.0,maxval=1.0)
        init_state = self.lstm_cell.zero_state(this_size, tf.float32)

        
        inputs=[self.gNet(self.X,start_location)]
        inputs.extend([0]*config.N_glimpse)
        self.loc_his=[start_location]
        self.loglikelihood_his=[]
        self.baseline_his=[]
        
        outputs,_ = rnn_decoder(inputs, init_state, self.lstm_cell, loop_function=self.get_next_input)
        lstm_output=outputs[-1]

        # pack data for calculation
        baseline_his_tf=tf.stack(self.baseline_his)
        loglikelihood_his_tf=tf.stack(self.loglikelihood_his)
        reduce_llh=tf.reduce_mean(loglikelihood_his_tf)

        # make prediction
        score=self.predict_net(inputs=lstm_output)
        prediction=tf.argmax(score,1)

        # calculate reward, do variance reduction and calculate reinforced loglikelihood
        reward=tf.cast(tf.equal(prediction,tf.argmax(self.y,1)),dtype=tf.float32)
        # stop gradient on reward(redundant because tf.equal does not have gradient)
        reward=tf.stop_gradient(reward)
        accuracy=tf.reduce_sum(reward)/tf.cast(this_size,dtype=tf.float32)
        reduce_var_reward=reward-tf.stop_gradient(baseline_his_tf)
        reinforce_llh=tf.reduce_mean(loglikelihood_his_tf*reduce_var_reward)

        # regression baseline towards reward
        baseline_mse=tf.reduce_mean(tf.square(reward-baseline_his_tf))

        # softmax to output
        softmax_loss=tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y,logits=score))

        # summarize loss
        loss=reinforce_llh+baseline_mse+softmax_loss

        # # for testing gradient flow
        # dweight1=tf.gradients(loss,[prediction])
        # dweight2=tf.gradients(reinforce_llh,[prediction])
        # dweight3=tf.gradients(baseline_mse,[prediction])
        # dweight4=tf.gradients(softmax_loss,[prediction])
        # print(dweight1,dweight2,dweight3,dweight4)

        return loss,accuracy
    
if __name__=="__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("../mnist/", one_hot=True)
    
    num_train=mnist.train.num_examples
    num_val=mnist.validation.num_examples
    num_test=mnist.test.num_examples
    
    ram=RAM()
    loss,accuracy=ram()
    
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 4e-3
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,num_train//config.batch_size, 0.95, staircase=True)
    optimizier=tf.train.RMSPropOptimizer(learning_rate=1e-3)
    train_step = optimizier.minimize(loss,global_step=global_step)
    
    max_epoch=50
    print_every=50

    def train():
        num_iteration=num_train//config.batch_size
        for it in range(num_iteration):
            images,labels=mnist.train.next_batch(config.batch_size)
            # prepare data for monte carlo test
            images=np.tile(images,(config.MC_test,1))
            labels=np.tile(labels,(config.MC_test,1))
            feed_dict={ram.X:images.reshape(config.tot_size,28,28,1),ram.y:labels}
            loss_num,acc_num,_ = sess.run([loss,accuracy,train_step],feed_dict=feed_dict)
            if it==0 or (it+1)%print_every==0 or it==num_iteration-1:
                print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                      'iteration %d/%d:' % (it+1,num_iteration),'loss=%8f, accuracy=%.3f%%' % (loss_num,acc_num*100.0))

    def eval(dataset,num_iteration):
        total_loss=0
        total_accuracy=0
        for it in range(num_iteration):
            images,labels=dataset.next_batch(config.batch_size)
            # no Monte Carlo test during evaludation step
            feed_dict={ram.X:images.reshape(config.batch_size,28,28,1),ram.y:labels}
            loss_num,accuracy_num = sess.run([loss,accuracy],feed_dict=feed_dict)
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
#         for a in tf.global_variables():
#             print('variable ',a)
        tf.global_variables_initializer().run()
        max_acc=None
        for epoch in range(max_epoch):
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),
                  'start epoch %d/%d, with learning rate = %f' % (epoch+1,max_epoch,sess.run(learning_rate)))
            train()
            loss_train,acc_train=eval(mnist.train,num_train//config.batch_size)
            loss_val,acc_val=eval(mnist.validation,num_val//config.batch_size)
            loss_test,acc_test=eval(mnist.test,num_test//config.batch_size)
            acc_train_his.append(acc_train)
            acc_val_his.append(acc_val)
            acc_test_his.append(acc_test)

            if max_acc==None or acc_val>max_acc:
                max_acc=acc_val
                save_path = saver.save(sess, "parameters/RAM/RAM.ckpt")
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
    plt.savefig('model-RAM.png')
    plt.show()
