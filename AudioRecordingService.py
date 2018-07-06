#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

from io import BytesIO
from io import StringIO
import pyaudio
import wave


FORMAT = pyaudio.paInt16
CHANNELS = 1
#tensorflow expects the audio to have a sampling rate of 16000 Hz. 
RATE = 16000 #44100
CHUNK = 1024
RECORD_SECONDS = 1
FREQUENCY = 261.63 

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
						rate=RATE, input=True,
						frames_per_buffer=CHUNK)

def Close():
    print ("finished recording")
    stream.stop_stream()
    stream.close()
    audio.terminate()

def Record():
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
    return wav_data