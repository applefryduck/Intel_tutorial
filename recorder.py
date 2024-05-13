import pyaudio
import wave
import threading

class Recorder:
    def __init__(self, chunk=1024, channels=1, rate=44100):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self.is_recording = False
        self.frames = []

    def start_recording(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)
        self.is_recording = True
        self.frames = []

        print("Recording started...")

        while self.is_recording:
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            print("Recording stopped.")
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            self.save_recording()


    def save_recording(self, filename="temp.wav"):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print(f"Recording saved to {filename}")