import os
import sys
import time
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from scipy.io.wavfile import read, write
from keras.models import Model, Sequential
from keras.layers import Convolution1D, Conv1D, Flatten, Dense, \
    Input, Lambda, multiply, add, Activation
from keras.models import load_model
from keras.callbacks import TensorBoard
import random

def wavenetBlock(n_atrous_filters, atrous_filter_size, atrous_rate):
    def f(input_):
        residual = input_
        tanh_out = Conv1D(n_atrous_filters, atrous_filter_size,
                                       padding='same',
                                       activation='tanh')(input_)
        sigmoid_out = Conv1D(n_atrous_filters, atrous_filter_size,
                                          padding='same',
                                          activation='sigmoid')(input_)
        merged = multiply([tanh_out, sigmoid_out])
        skip_out = Convolution1D(1, 1, activation='relu', padding='same')(merged)
        out = add([skip_out, residual])
        return out, skip_out
    return f


def get_basic_generative_model(input_size):
    input_ = Input(shape=(input_size, 1))
    A, B = wavenetBlock(64, 2, 2)(input_)
    skip_connections = [B]
    for i in range(20):
        A, B = wavenetBlock(64, 2, 2**((i+2)%9))(A)
        skip_connections.append(B)
    net = add(skip_connections)
    net = Activation('relu')(net)
    net = Convolution1D(1, 1, activation='relu')(net)
    net = Convolution1D(1, 1)(net)
    net = Flatten()(net)
    net = Dense(256, activation='softmax')(net)
    model = Model(input=input_, output=net)
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def get_audio(filename):
    sr, audio = read(filename)
    audio = audio.astype(float)
    audio = audio - audio.min()
    audio = audio / (audio.max() - audio.min())
    audio = (audio - 0.5) * 2
    return sr, audio

def frame_generator(sr, audio, frame_size, frame_shift, minibatch_size=20):
    '''
    :param sr            : sample rate
    :param audio         : audio data
    :param frame_size    : 2048 fixed
    :param frame_shift   : 128 fixed
    :param minibatch_size: 20 default
    :return              : np.array(X), np.array(y)
    '''
    audio_len = len(audio)
    X = []
    y = []
    while 1:
        #print("frame_generator: while loop sr=" + str(sr) + " frame_size=" + str(frame_size) + " frame_shift=" + str(frame_shift) + " minibatch_size=" + str(minibatch_size) )
        end = audio_len - frame_size - 1
        start = random.randint(0, end)
        for i in range(start, end, frame_shift):
            frame = audio[i:i+frame_size]
            if len(frame) < frame_size:
                break
            if i + frame_size >= audio_len:
                break
            temp = audio[i + frame_size]
            target_val = int((np.sign(temp) * (np.log(1 + 256*abs(temp)) / (
                np.log(1+256))) + 1)/2.0 * 255)
            X.append(frame.reshape(frame_size, 1))
            y.append((np.eye(256)[target_val]))
            if len(X) == minibatch_size:
                #print(" i=" + str(i) + "/" + str(end))
                yield np.array(X), np.array(y)
                X = []
                y = []


def get_audio_from_model(model, sr, duration, seed_audio):
    print('Generating audio...')
    new_audio = np.zeros(int(sr * duration))
    curr_sample_idx = 0
    while curr_sample_idx < new_audio.shape[0]:
        distribution = np.array(model.predict(seed_audio.reshape(1,
                                                                 frame_size, 1)
                                             ), dtype=float).reshape(256)
        distribution /= distribution.sum().astype(float)
        predicted_val = np.random.choice(range(256), p=distribution)
        ampl_val_8 = ((((predicted_val) / 255.0) - 0.5) * 2.0)
        ampl_val_16 = (np.sign(ampl_val_8) * (1/256.0) * ((1 + 256.0)**abs(
            ampl_val_8) - 1)) * 2**15
        new_audio[curr_sample_idx] = ampl_val_16
        seed_audio[:-1] = seed_audio[1:]
        seed_audio[-1] = ampl_val_16
        pc_str = str(round(100*curr_sample_idx/float(new_audio.shape[0]), 2))
        sys.stdout.write('Percent complete: ' + pc_str + '\r')
        sys.stdout.flush()
        curr_sample_idx += 1
    print('Audio generated.')
    return new_audio.astype(np.int16)


class SaveAudioCallback(Callback):
    def __init__(self, ckpt_freq, sr, seed_audio, output_dir):
        super(SaveAudioCallback, self).__init__()
        self.ckpt_freq = ckpt_freq
        self.sr = sr
        self.seed_audio = seed_audio
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1)%self.ckpt_freq==0:
            ts = str(int(time.time()))
            filepath = self.output_dir + '/ckpt_' + ts + '.wav'
            audio = get_audio_from_model(self.model, self.sr, 0.5, self.seed_audio)
            write(filepath, self.sr, audio)

def mkdir_if_needed(dir):
  if not os.path.isdir(dir):
      os.mkdir(dir)

def get_saved_model(dir):
  files = glob.glob(os.path.join(dir, '*.h5'))
  if files:
    model = max(files, key=os.path.getmtime)
    print("load model from " + model + "...")
    return load_model(model)
  else:
    return None

def generate_wav(model, sr, audio_context, output_dir, str_timestamp):
  new_audio = get_audio_from_model(model, sr, 2, audio_context)
  outfilepath = output_dir+'/generated_'+str_timestamp+'.wav'
  print('Writing generated audio to:', outfilepath)
  write(outfilepath, sr, new_audio)

if __name__ == '__main__':
    n_epochs = 30
    frame_size = 2048
    frame_shift = 128
    sr_training, training_audio = get_audio('train.wav')
    # training_audio = training_audio[:sr_training*1200]
    sr_valid, valid_audio = get_audio('validate.wav')
    # valid_audio = valid_audio[:sr_valid*60]
    assert sr_training == sr_valid, "Training, validation samplerate mismatch"
    n_training_examples = int((len(training_audio)-frame_size-1) / float(
        frame_shift))
    n_validation_examples = int((len(valid_audio)-frame_size-1) / float(
        frame_shift))
    print('Total training examples:', n_training_examples)
    print('Total validation examples:', n_validation_examples)

    output_dir = 'output'
    models_dir = 'models'
    logs_dir   = 'models'
    mkdir_if_needed(output_dir)
    mkdir_if_needed(models_dir)
    mkdir_if_needed(logs_dir)

    model = get_saved_model(models_dir)
    if not model:
      model = get_basic_generative_model(frame_size)
    audio_context = valid_audio[:frame_size]
    save_audio_clbk = SaveAudioCallback(100, sr_training, audio_context, output_dir)
    tensorboard_clbk = TensorBoard(log_dir='./logs')
    validation_data_gen = frame_generator(sr_valid, valid_audio, frame_size, frame_shift)
    training_data_gen = frame_generator(sr_training, training_audio, frame_size, frame_shift)
    hist = model.fit_generator(training_data_gen, 
      steps_per_epoch=60, 
      epochs=n_epochs, 
      validation_data=validation_data_gen,
      validation_steps=5, 
      verbose=1, 
      callbacks=[save_audio_clbk, tensorboard_clbk])

    str_timestamp = str(int(time.time()))
    outfilepath = models_dir+'/model_'+str_timestamp+'_'+str(n_epochs)+'.h5'
    print('Saving model to:' + outfilepath)
    model.save(outfilepath)

    #generate_wav(model, sr_training, audio_context, output_dir, str_timestamp)
    print('\nDone!')
