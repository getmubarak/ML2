#import sys
import tensorflow as tf
# pylint: enable=unused-import
#load AI modules
import AudioRecognitionService
import AudioRecordingService

def label_wav(labels, graph, input_name, output_name, how_many_labels):
  """Loads the model and labels, and runs the inference to print predictions."""
  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = AudioRecognitionService.load_labels(labels)

  # load graph, which is stored in the default session
  AudioRecognitionService.load_graph(graph)
  #AudioRecordingService.Open()
  while(1):
      wav_data = AudioRecordingService.Record()
      command = AudioRecognitionService.run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)
      print(command[0])
      print(command[1])

  AudioRecordingService.Close()

if __name__ == '__main__':
  how_many_labels = 1
  input_name = 'wav_data:0'
  output_name = 'labels_softmax:0'
  label_wav('model/speech_commands_train/conv_labels.txt', 'model/my_frozen_graph.pb', input_name,
            output_name, how_many_labels)
