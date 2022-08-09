#importing required libraries
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import boto3           #libary for accessing aws
import json            #library for converting data into json format for sending any message on web
import time
import numpy as np
import pyaudio
import pandas as pd

client = boto3.client('kinesis')
stream_name='audio_data'

#adding a button
st.title("Speech Emotion Recognizer")
st.text('Please record some audio to detect emotion')

def record(duration=3, fs=8000):
    nsamples = duration*fs
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True,
                    frames_per_buffer=nsamples)
    buffer = stream.read(nsamples)
    array = np.frombuffer(buffer, dtype='int16')
    stream.stop_stream()
    stream.close()
    p.terminate()
    return array

if st.button('Record Audio'):
    #Hardcoded to send 10 audio clips each time
    st.write("Recording audio ....")
    #Recording a 3 second audio clip
    count = 0
    while count<10:
        my_recording = record()
        fs=8000
        # seconds = 5  # Duration of recording
        # myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        write('output.wav', fs, my_recording)  # Save as WAV file 
        #time.sleep(5)
        y, sr = librosa.load('output.wav')
        # Extract the MFCC features from audio file
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        # Extracting the mean and standard deviation of the MFCC features across time
        features = pd.Series(np.hstack((np.mean(mfcc, axis=1), np.std(mfcc, axis=1))))
        message = {
            'user_id':np.random.randint(10),
            'message_type':'audio',
            'mfcc': features.to_json()
            
        }
        #Pushing the data record into the kinesis stream
        client.put_record(
            StreamName="audio_data",
            Data=json.dumps(message),
            PartitionKey='user_id'
        )
        count+=1
    #Sending audio clip to stream
    st.write("Sending audio to kinesis stream ...")
    st.write("Done Recording audio.")
#Container to hold output sentiment
# with st.empty():
#  for seconds in range(60):
#      st.write(f"⏳ {seconds} seconds have passed")
#      time.sleep(1)
#  st.write("✔️ 1 minute over!")

