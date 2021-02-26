from multiprocessing import Queue
from multiprocessing.managers import BaseManager
from job import Job
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import os
from keras.models import model_from_json
import tensorflow as tf

dispatched_job_queue = Queue()
finished_job_queue = Queue()
def get_finished_job_queue():
    return finished_job_queue
def get_dispatched_job_queue():
    return dispatched_job_queue
class Server:
    def __init__(self):
        BaseManager.register('get_dispatched_job_queue', callable = get_dispatched_job_queue)
        BaseManager.register('get_finished_job_queue', callable = get_finished_job_queue)
        tf.app.flags.DEFINE_string("address", "", "Server Address")
        FLAGS = tf.app.flags.FLAGS
        if FLAGS.address == "":
            self.manager = BaseManager(address=('210.70.145.14', 5000), authkey = b'jobs')
        else:
            self.manager = BaseManager(address=(FLAGS.address, 5000), authkey = b'jobs')
        
    def cifar10_data(self):
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        x_train = X_train.astype('float32')/255
        x_test = X_test.astype('float32')/255
        y_train = np_utils.to_categorical(Y_train)
        y_test = np_utils.to_categorical(Y_test)
        return x_train, x_test, y_train, y_test

    def model_cdnn(self):
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=3, input_shape=(32, 32, 3), activation='relu', padding='same'))
        model.add(Conv2D(filters=64, kernel_size=3, input_shape=(32, 32, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2))

        model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2))

        model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=2))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(rate=0.25))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    

    def model_information(self, job, model_json, loss, optimizer, metrics, job_data):
        job.model_con = model_json
        job.loss = loss
        job.optimizer = optimizer
        job.metrics = metrics
        job.data = job_data

    def start(self):
        self.manager.start()

        dispatched_jobs = self.manager.get_dispatched_job_queue()
        finished_jobs = self.manager.get_finished_job_queue()
        #build model
        model = self.model_cdnn()
        json_string = model.to_json()
        
        job = Job(0)
        #traing data input job.data 
        x_train, x_test, y_train, y_test = self.cifar10_data()
        job_data = (x_train, x_test, y_train, y_test)
        #'''transport datas to job'''
        self.model_information(job, json_string, loss = 'categorical_crossentropy', optimizer = 'adam', metrics = 'accuracy',job_data = job_data)

        dispatched_jobs.put(job)
        print('dispathced job: %s', job.job_id)
        
        while not dispatched_jobs.empty() or not finished_jobs.empty():
            job = finished_jobs.get(60)
            model.set_weights(job.weights)
            model.save_weights('models.h5')
            print('save weights done!!')
            print('Finished Job: %s', job.job_id)
            print('Test:')
            print('Loss:', job.info[0])
            print('Accuracy:', job.info[1])

if __name__ == "__main__":
    Server = Server()
    Server.start()