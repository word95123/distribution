import time
from multiprocessing import Queue
from multiprocessing.managers import BaseManager
from job import Job
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import model_from_json
import tensorflow as tf
class Slave:
    def __init__(self):
        self.dispatched_job_queue = Queue()
        self.finished_job_queue = Queue()
        tf.app.flags.DEFINE_string("address", "", "Client Address")
        self.FLAGS = tf.app.flags.FLAGS

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

    

    def start(self):
        
        BaseManager.register('get_dispatched_job_queue')
        BaseManager.register('get_finished_job_queue')
        
        if self.FLAGS.address == "":
            server = '210.70.145.14'
            print('Connnect to server %s...' % server)
            manager = BaseManager(address=(server, 5000), authkey = b'jobs')
        else:
            print('Connnect to server %s...' % self.FLAGS.address)
            manager = BaseManager(address=(self.FLAGS.address, 5000), authkey = b'jobs')

        manager.connect()
        
        
        
        

        dispatched_jobs = manager.get_dispatched_job_queue()
        finished_jobs = manager.get_finished_job_queue()
        #model = self.model_cdnn()
        
        while not dispatched_jobs.empty():
            job = dispatched_jobs.get(timeout = 1)
            json_string = job.model_con
            model = Sequential()
            model = model_from_json(json_string)
            model.compile(loss=job.loss, optimizer=job.optimizer, metrics=[job.metrics])
            model.fit(job.data[0], job.data[2], epochs=1, batch_size=128, verbose=1)
            weights = model.get_weights()
            loss, accuracy = model.evaluate(job.data[1], job.data[3])
            job.weights = weights
            job.info = (loss, accuracy)
            print('Run job: %s' % job.job_id)
            

            time.sleep(1)
            finished_jobs.put(job)
if __name__ == "__main__":
    slave = Slave()
    slave.start()