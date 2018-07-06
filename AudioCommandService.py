from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from io import BytesIO
from io import StringIO
import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import
import pyaudio
import wave

FORMAT = pyaudio.paInt16
CHANNELS = 1
#tensorflow expects the audio to have a sampling rate of 16000 Hz. 
RATE = 16000 #44100
CHUNK = 1024
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "file.wav"
FREQUENCY = 261.63 

audio = pyaudio.PyAudio()

FLAGS = None


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    return 0


def label_wav(labels, graph, input_name, output_name, how_many_labels):
  """Loads the model and labels, and runs the inference to print predictions."""
  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)
  stream = audio.open(format=FORMAT, channels=CHANNELS,
						rate=RATE, input=True,
						frames_per_buffer=CHUNK)
  while(1):
      print ("recording...")
      frames = []
      for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

   
      memory_file = BytesIO()
      #memory_file  = StringIO(wav_buffer)
      waveFile = wave.open(memory_file, 'wb')
      waveFile.setnchannels(CHANNELS)
      waveFile.setsampwidth(audio.get_sample_size(FORMAT))
      waveFile.setframerate(RATE)
      waveFile.writeframes(b''.join(frames))
      waveFile.close()
      memory_file.flush()
      wav_data = memory_file.getvalue()
      run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)

  print ("finished recording")
  #stop Recording
  stream.stop_stream()
  stream.close()
  audio.terminate()

if __name__ == '__main__':
  how_many_labels = 1
  input_name = 'wav_data:0'
  output_name = 'labels_softmax:0'
  label_wav('model/speech_commands_train/conv_labels.txt', 'model/my_frozen_graph.pb', input_name,
            output_name, how_many_labels)
