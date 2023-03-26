# trying to make a sweep deconvolution script for ambisonic microphones (4 channels until the moment)
# building on https://gist.github.com/josephernest/
# and my own code, 
# enrique tomas
# September 2022-2023

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt 
import wave
from pydub import AudioSegment
import wavfile                  #a custom script to be copied to the same folder

# AUX functions 
def ratio(dB):
    return np.power(10, dB * 1.0 / 20)

def padarray(A, length, before=0):
    t = length - len(A) - before
    if t > 0:
        width = (before, t) if A.ndim == 1 else ([before, t], [0, 0])
        return np.pad(A, pad_width=width, mode='constant')
    else:
        width = (before, 0) if A.ndim == 1 else ([before, 0], [0, 0])
        return np.pad(A[:length - before], pad_width=width, mode='constant')

def filter20_20k(x, sr): # filters everything outside out 20 - 20000 Hz
    nyq = 0.5 * sr
    sos = signal.butter(5, [20.0 / nyq, 20000.0 / nyq], btype='band', output='sos')
    return signal.sosfilt(sos, x)

#extracts one channel from a multichannel file and saves it to a mono file
#usage: 
# wav = wave.open(WAV_FILENAME) 
# separate_wav_channel('ch1.wav', wav, 0) #(first channel)
# separate_wav_channel('ch1.wav', wav, 2) #(third channel)

def extract_wav_channel(fn, wav, channel):
    '''
    Take Wave_read object as an input and save one of its
    channels into a separate .wav file.
    '''
    # Read data
    nch   = wav.getnchannels()
    depth = wav.getsampwidth()
    wav.setpos(0)
    sdata = wav.readframes(wav.getnframes())

    # Extract channel data (24-bit data not supported)
    typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
    if not typ:
        raise ValueError("sample width {} not supported".format(depth))
    if channel >= nch:
        raise ValueError("cannot extract channel {} out of {}".format(channel+1, nch))
    #print ("Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, nch, depth*8))
    data = np.fromstring(sdata, dtype=typ)
    ch_data = data[channel::nch]

    # Save channel to a separate file
    outwav = wave.open(fn, 'w')
    outwav.setparams(wav.getparams())
    outwav.setnchannels(1)
    outwav.writeframes(ch_data.tostring())
    outwav.close()
    
def process(SWEEPFILE, wav_file, OUTFILE, channels_nr):
    
    #SWEEP VISUALISATION
    #sweep_visualisation(SWEEPFILE)
    
    #first 4ch channel extraction to mono files

    WAV_FILENAME = wav_file
    wav = wave.open(WAV_FILENAME)
    
    if channels_nr == 1:
        h, sr = calculate_IR(SWEEPFILE, wav_file, OUTFILE)
        visualise_IR(h,sr)
        print("Exported mono IR file {} " .format(OUTFILE))
    if channels_nr == 2:
        extract_wav_channel('ch1.wav', wav, 0)
        h1, sr1 = calculate_IR(SWEEPFILE, 'ch1.wav', 'ir1.wav')
        visualise_IR(h1,sr1)
        extract_wav_channel('ch2.wav', wav, 1)
        h2, sr2 = calculate_IR(SWEEPFILE, 'ch2.wav', 'ir2.wav')
        visualise_IR(h2,sr2)
        
        # load individual channels and create a stereo IR file
        left_channel = AudioSegment.from_wav("ir1.wav")
        right_channel = AudioSegment.from_wav("ir2.wav")

        stereo_sound = AudioSegment.from_mono_audiosegments(left_channel, right_channel)
        # simple export
        file_handle = stereo_sound.export(OUTFILE, format="wav")
        
        print("Exported IR stereo file {} " .format(OUTFILE))
        #print ("Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, nch, depth*8))
        
    if channels_nr == 4:
        extract_wav_channel('ch1.wav', wav, 0)
        h1, sr1 = calculate_IR(SWEEPFILE, 'ch1.wav', 'ir1.wav')
        visualise_IR(h1,sr1)
        extract_wav_channel('ch2.wav', wav, 1)
        h2, sr2 = calculate_IR(SWEEPFILE, 'ch2.wav', 'ir2.wav')
        visualise_IR(h2,sr2)
        extract_wav_channel('ch3.wav', wav, 2)
        h3, sr3 = calculate_IR(SWEEPFILE, 'ch3.wav', 'ir3.wav')
        visualise_IR(h3,sr3)
        extract_wav_channel('ch4.wav', wav, 3)
        h4, sr4 = calculate_IR(SWEEPFILE, 'ch4.wav', 'ir4.wav')
        visualise_IR(h4,sr4)

        # load individual channels and create a multichannel IR file
        first_channel = AudioSegment.from_wav("ir1.wav")
        second_channel = AudioSegment.from_wav("ir2.wav")
        third_channel = AudioSegment.from_wav("ir3.wav")
        fourth_channel = AudioSegment.from_wav("ir4.wav")

        four_ch_sound = AudioSegment.from_mono_audiosegments(first_channel, second_channel, third_channel, fourth_channel)
        # simple export
        file_handle = four_ch_sound.export(OUTFILE, format="wav")
        print("Exported 4channels IR file {} " .format(OUTFILE))

def calculate_IR(SWEEPFILE, RECFILE, OUTFILE):  #per mono file

    #load previous files and get sample rate, bitrate and data arrays "a" and "b"
    sr, a, br = wavfile.read(SWEEPFILE, normalized=True)
    sr, b, br = wavfile.read(RECFILE, normalized=True)
    
    # Deconvolution 
    # a is the input sweep signal, h the impulse response, and b the microphone-recorded signal. 
    # We have a * h = b (convolution here!). 
    # Let's take the discrete Fourier transform, we have fft(a) * fft(h) = fft(b), 
    # then h = ifft(fft(b) / fft(a)).

    a = padarray(a, sr*50, before=sr*10)
    b = padarray(b, sr*50, before=sr*10)
    h = np.zeros_like(b)

    h = np.zeros_like(b)

    b1 = filter20_20k(b, sr)

    ffta = np.fft.rfft(a)

    fftb = np.fft.rfft(b1)
    ffth = fftb / ffta
    h1 = np.fft.irfft(ffth)

    h1 = filter20_20k(h1, sr)

    h = h1

    h = h[:10 * sr]
    h *= ratio(dB=40)

    #write to file
    wavfile.write(OUTFILE, sr, h, normalized=True, bitrate=16)
    
    return h, sr

def visualise_IR(hh,sr):    
    #VISUALIZE IR (mono)

    #we need to extract sample rate and number of samples to work with the file
    n_samples = hh.size

    #calculate duration of the file
    t_audio = n_samples/sr

    #Plot the sweep
    # 1. create an array for the x axis - time with the exact time of each sample
    times = np.linspace(0, n_samples/sr, num=round(n_samples)) #if 16 bits
    #print("size of time axis array:", times.size)
    #print("size of sweep axis array:", h.size)

    #plot
    plt.figure(figsize=(15, 5))
    plt.plot(times, hh)
    plt.title('IR')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio/2)
    plt.show()
    
def sweep_visualisation(SWEEPFILE):
    #load previous files and get sample rate, bitrate and data arrays "a" and "b"
    sr, a, br = wavfile.read(SWEEPFILE, normalized=True)
    
    #VISUALIZE ORIGINAL SWEEP (mono)


    #we need to extract sample rate and number of samples to work with the file
    n_samples = a.size

    #calculate duration of the file
    t_audio = n_samples/sr

    #Plot the sweep
    # 1. create an array for the x axis - time with the exact time of each sample
    times = np.linspace(0, n_samples/sr, num=round(n_samples)) #if 16 bits
    #times = np.linspace(0, n_samples/sample_freq, num=round(n_samples/2)) #num=round(n_samples/2)) if 32 bits
    print("size of time axis array:", times.size)
    print("size of sweep axis array:", a.size)

    #plot
    plt.figure(figsize=(15, 5))
    plt.plot(times, a)
    plt.title('Original Sweep')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    plt.show()
    
    # trim the first 25 seconds
    samples = a[:int(sr*25)]
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(samples, Fs=sr)
    plt.show()  