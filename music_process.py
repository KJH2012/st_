import os
import zipfile
import shutil
import urllib.request
import numpy as np
import librosa
import soundfile as sf
import moviepy.editor as mpe
from scipy.signal import welch
from pathlib import Path
import psola
import scipy.signal as sig
import sys
sys.path.append('./Python-Midi-Analysis-master')
from MidiData import MidiData
from scipy.signal import lfilter
from numpy import pi, sin, tan, linspace, mod, clip, sqrt, abs

def normalize_audio(audio):
    return audio / np.max(np.abs(audio))

def calculate_segment_dB(segment, sampling_rate):
    f, Pxx = welch(segment, fs=sampling_rate, nperseg=len(segment))
    db = 10 * np.log10(Pxx)
    return db

def analyze_and_save_voice_file(audio_file_path, video_file_path, folder_path, backing_file_path, first_audio_output_filename):
    y, sr = librosa.load(audio_file_path, sr=None, mono=True)
    y_normalized = normalize_audio(y)

    segment_duration = 0.2
    segment_length = int(segment_duration * sr)

    start_time = 0
    video_clip = mpe.VideoFileClip(video_file_path)
    end_time = video_clip.duration
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    num_segments = (end_sample - start_sample) // segment_length

    db_results = []

    for i in range(num_segments):
        start_idx = start_sample + i * segment_length
        end_idx = start_idx + segment_length

        segment = y_normalized[start_idx:end_idx]
        db_mean = np.mean(calculate_segment_dB(segment, sr))
        db_results.append(db_mean)

    time_ranges = []
    curr_range = [0, None, None]
    curr_state = "Y" if db_results[0] >= -110 else "N"

    for i in range(1, num_segments):
        state = "Y" if db_results[i] >= -110 else "N"
        if state != curr_state:
            curr_range[1] = i * segment_duration
            curr_range[2] = curr_state
            time_ranges.append(tuple(curr_range))
            curr_state = state
            curr_range = [i * segment_duration, None, None]

    if curr_range[1] is None:
        curr_range[1] = num_segments * segment_duration
        curr_range[2] = curr_state
        time_ranges.append(tuple(curr_range))

    audio_segments_to_concat = []
    video_segments_to_concat = []

    for idx, time_range in enumerate(time_ranges):
        if time_range[2] == "Y":
            audio_segment = y_normalized[int(time_range[0]*sr):int(time_range[1]*sr)]
            video_segment = video_clip.subclip(time_range[0], time_range[1])
            
            audio_segments_to_concat.append(audio_segment)
            video_segments_to_concat.append(video_segment)

    combined_audio = np.concatenate(audio_segments_to_concat)
    combined_video = mpe.concatenate_videoclips(video_segments_to_concat)
    combined_video = combined_video.resize((720, 1280))

    backing_y, backing_sr = librosa.load(backing_file_path, sr=None, mono=True)
    backing_duration = len(backing_y) / backing_sr
    
    num_repeats = int(np.ceil(backing_duration / combined_video.duration))

    combined_audio = np.tile(combined_audio, num_repeats)
    combined_video_clips = [combined_video] * num_repeats
    final_combined_video = mpe.concatenate_videoclips(combined_video_clips)

    sf.write(os.path.join(folder_path, first_audio_output_filename), combined_audio, sr)    
    final_combined_video.write_videofile(os.path.join(folder_path, 'first_video.mp4'))

    video_clip.close()
    final_combined_video.close()


# Convert a MIDI note number to its frequency in Hz
def midi_note_to_frequency(note_number):
    return 440.0 * (2.0**((note_number - 69) / 12.0))

# Analyze the MIDI file and extract note data
def analyze_midi_file(midi_file_path):
    midi_data = MidiData(midi_file_path)
    notes_data = []
    for track_index in range(midi_data.get_num_tracks()):
        track = midi_data.get_track(track_index)
        for note in track.notes:
            note_data = {
                "start_time": note.start_time,
                "end_time": note.end_time,
                "note_number": note.pitch,
                "frequency": midi_note_to_frequency(note.pitch)
            }
            notes_data.append(note_data)
    return notes_data

'''33 frequencies -> Bandpass filter design coefficients'''

a = [
    [1,	-3.99975599432850,	5.99927319549360,	-3.99927840737790,	0.999761206219333],
    [1,	-3.99969087479883,	5.99908089886356,	-3.99908917207623,	0.999699148027989],
    [1	-3.99960783242349,	5.99883663242320,	-3.99884976505012,	0.999620965091944],
    [1,	-3.99950162319079,	5.99852572084418,	-3.99854656702148,	0.999522469472722],
    [1,	-3.99936529624421,	5.99812898911301,	-3.99816207919623,	0.999398386591067],
    [1,	-3.99918954838096,	5.99762119086212,	-3.99767371571456,	0.999242073897684],
    [1,	-3.99896179230423,	5.99696879242825,	-3.99705216552381,	0.999045167073510],
    [1,	-3.99866479474709,	5.99612680715249,	-3.99625914349455,	0.998797135306096],
    [1,	-3.99827466248870,	5.99503421446511,	-3.99524426396985,	0.998484722617774],
    [1,	-3.99775783209982,	5.99360724998732,	-3.99394063771490,	0.998091246593493],
    [1,	-3.99706652768996,	5.99172946648071,	-3.99225859033826,	0.997595718976056],
    [1,	-3.99613185013489,	5.98923686336783,	-3.99007658764110,	0.996971744260045],
    [1,	-3.99485318855908,	5.98589544431097,	-3.98722797037749,	0.996186142439803],
    [1,	-3.99308190186349,	5.98136710272466,	-3.98348135377444,	0.995197230334171],
    [1,	-3.99059605102152,	5.97515747320379,	-3.97851139167541,	0.993952682453192],
    [1,	-3.98706113267000,	5.96653590681851,	-3.97185482190872,	0.992386877515924],
    [1,	-3.98196890312390,	5.95441243992289,	-3.96284396965905,	0.990417621333426],
    [1,	-3.97454193106546,	5.93714875785070,	-3.95050570870629,	0.987942122629232],
    [1,	-3.96358465094425,	5.91226887379876,	-3.93340759291814,	0.984832088850899],
    [1,	-3.94725123836206,	5.87602016123001,	-3.90942359895478,	0.980927810114349],
    [1,	-3.92268507313800,	5.82271809701412,	-3.87537871287358,	0.976031121355393],
    [1,	-3.88546228104775,	5.74379634972886,	-3.82651390292709,	0.969897192591698],
    [1,	-3.82874219179635,	5.62650200982447,	-3.75569214621489,	0.962225222699710],
    [1,	-3.74199398552012,	5.45228903999113,	-3.65224899613469,	0.952648347632671],
    [1,	-3.60914741639923,	5.19532970510166,	-3.50039918643988,	0.940723488871680],
    [1,	-3.40605615180314,	4.82253055457138,	-3.27720061176920,	0.925922567908390],
    [1,	-3.09739763001925,	4.29865336456317,	-2.95038375754678,	0.907627654519980],
    [1,	-2.63388305596713,	3.60440511182782,	-2.47717214208704,	0.885134424906144],
    [1,	-1.95260897719969,	2.78102389972874,	-1.80711132908444,	0.857671089645940],
    [1,	-0.987821704356849,	2.01302475585146,	-0.895725041038555,	0.824444128755633],
    [1,	0.292204692740340,	1.71579119867574,	0.258147856757345,	0.784728343490888],
    [1,	1.79567000881120,	2.42888468072076,	1.53417945578454,	0.738028006232103],
    [1,	3.14391651047433,	4.05911136375890,	2.57181632513949,	0.684351098394969]
    ]

b = [
    [1.71790337413308e-08,	0,	-3.43580674826616e-08,	 0,	     1.71790337413308e-08],
    [2.72392949892633e-08,	0,	-5.44785899785266e-08,	 0,	     2.72392949892633e-08],
    [4.33202037600520e-08,	0,	-8.66404075201039e-08,	 0,	     4.33202037600520e-08],
    [6.87019970842440e-08,	0,	-1.37403994168488e-07,	 0,	     6.87019970842440e-08],
    [1.09076437679077e-07,	0,	-2.18152875358153e-07,	 0,	     1.09076437679077e-07],
    [1.73127013142121e-07,	0,	-3.46254026284243e-07,	 0,	     1.73127013142121e-07],
    [2.74802658915039e-07,	0,	-5.49605317830078e-07,	 0,	     2.74802658915039e-07],
    [4.36167681516868e-07,	0,	-8.72335363033737e-07,	 0,	     4.36167681516868e-07],
    [6.92263017611530e-07,	0,	-1.38452603522306e-06,	 0,	     6.92263017611530e-07],
    [1.09868242572049e-06,	0,	-2.19736485144099e-06,	 0,	     1.09868242572049e-06],
    [1.74361621177647e-06,	0,	-3.48723242355294e-06,	 0,	     1.74361621177647e-06],
    [2.76695320840727e-06,	0,	-5.53390641681454e-06,	 0,	     2.76695320840727e-06],
    [4.39053611412317e-06,	0,	-8.78107222824634e-06,	 0,	     4.39053611412317e-06],
    [6.96608739369304e-06,	0,	-1.39321747873861e-05,	 0,	     6.96608739369304e-06],
    [1.10510729592236e-05,	0,	-2.21021459184473e-05,	 0,	     1.10510729592236e-05],
    [1.75286996737880e-05,	0,	-3.50573993475760e-05,	 0,	     1.75286996737880e-05],
    [2.77975505699513e-05,	0,	-5.55951011399026e-05,	 0,	     2.77975505699513e-05],
    [4.40709217599428e-05,	0,	-8.81418435198856e-05,	 0,	     4.40709217599428e-05],
    [6.98486316325863e-05,	0,	-0.000139697263265173,	 0,	     6.98486316325863e-05],
    [0.000110659311323076,	0,	-0.000221318622646152,	 0,	     0.000110659311323076],
    [0.000175225540477838,	0,	-0.000350451080955677,	 0,	     0.000175225540477838],
    [0.000277287390711482,	0,	-0.000554574781422964,	 0,	     0.000277287390711482],
    [0.000438446036225742,	0,	-0.000876892072451484,	 0,	     0.000438446036225742],
    [0.000692577807844958,	0,	-0.00138515561568992,	 0,	     0.000692577807844958],
    [0.00109264697016785,	0,	-0.00218529394033570,	 0,	     0.00109264697016785],
    [0.00172114638070978,	0,	-0.00344229276141955,	 0,	     0.00172114638070978],
    [0.00270595858002333,	0,	-0.00541191716004666,	 0,	     0.00270595858002333],
    [0.00424419670029157,	0,	-0.00848839340058314,	 0,	     0.00424419670029157],
    [0.00663759117688310,	0,	-0.0132751823537662,	 0,	     0.00663759117688310],
    [0.0103442686624309,    0,	-0.0206885373248617,	 0,	      0.0103442686624309],
    [0.0160534029877771,    0,	-0.0321068059755541,	 0,	      0.0160534029877771],
    [0.0247915230726592,    0,	-0.0495830461453185,	 0,	      0.0247915230726592],
    [0.0380733762023088,    0,	-0.0761467524046175,	 0,	      0.0380733762023088]
    ]

r = 0.99
lowpassf = [1.0, -2.0*r, +r*r]
d = 0.41004238851988095

#you can change f0 to new frequency hertz
f0=320
amp=1.0
Fs=22000
w=2.0*pi*f0/Fs
dB=10**(40/20)
chunk=4096
phase0=0
phase1=0
phase2=0
phase3=0
phase4=0
phase5=0
phase6=0
phase7=0


def carrier(s):
        global phase0
        global phase1
        global phase2
        global phase3
        global phase4
        global phase5
        global phase6
        global phase7

        #make It always continuos :-)

        '''
        phase0=w*(linspace(0, s, s))+phase0
        carriersignal= (amp*sawtooth(phase0))
        phase0 = mod(phase0[s-1], 2.0*pi);
        '''
        phase1=0.2 * w * (linspace(0, s, s)) + phase1
        phase2=0.4 * w * (linspace(0, s, s)) + phase2
        phase3=0.5 * w * (linspace(0, s, s)) + phase3
        phase4=2.0 * w * (linspace(0, s, s)) + phase4
        phase5=sin(phase1) - tan(phase3)
        phase6=sin(phase1) + sin(phase4)
        phase7=sin(phase2) - sin(phase4)
        x = sin(phase5)
        y = sin(phase6)
        z = sin(phase7)
        carriersignal = 0.25 * (x + y + z + d)
        phase1 = mod(phase1[s-1], 2.0*pi)
        phase2 = mod(phase2[s-1], 2.0*pi)
        phase3 = mod(phase3[s-1], 2.0*pi)
        phase4 = mod(phase4[s-1], 2.0*pi)
        phase5 = mod(phase5[s-1], 2.0*pi)
        phase6 = mod(phase6[s-1], 2.0*pi)
        phase7 = mod(phase7[s-1], 2.0*pi)
        
        #'''
        return carriersignal

def vocoder(sig):
        N=len(sig)

def carrier(s, w, d):
    phase1=0.2 * w * (linspace(0, s, s))
    phase2=0.4 * w * (linspace(0, s, s))
    phase3=0.5 * w * (linspace(0, s, s))
    phase4=2.0 * w * (linspace(0, s, s))
    phase5=sin(phase1) - tan(phase3)
    phase6=sin(phase1) + sin(phase4)
    phase7=sin(phase2) - sin(phase4)
    x = sin(phase5)
    y = sin(phase6)
    z = sin(phase7)
    carriersignal = 0.25 * (x + y + z + d)
    return carriersignal


def vocoder(sig, f0, w, a, b, lowpassf, d, dB, chunk):
    N = len(sig)
    carriersignal = carrier(N, w, d)
    vout = 0
    for i in range(0, 33):
        bandpasscarrier = lfilter(b[i], a[i], carriersignal)
        bandpassmodulator = lfilter(b[i], a[i], sig)
        rectifiedmodulator = abs(bandpassmodulator*bandpassmodulator)/N
        envelopemodulator = sqrt(lfilter([1.0], lowpassf, rectifiedmodulator))
        vout += bandpasscarrier*envelopemodulator
    vout = clip(vout*dB, -1, 1)
    return vout


# Integrated function for vocoder processing based on MIDI data
def simple_segment_audio_with_vocoder_integration(midi_file_path, first_audio_file_path, lowpassf, d, dB, chunk):
    midi_data = analyze_midi_file(midi_file_path)
    audio, sr = librosa.load(first_audio_file_path, sr=None)
    segments = []
    for note_data in midi_data:
        start_sample = int(note_data["start_time"] * sr / 1000)
        end_sample = int(note_data["end_time"] * sr / 1000)
        segment = audio[start_sample:end_sample]
        f0 = midi_note_to_frequency(note_data["note_number"])
        w = 2.0 * np.pi * f0 / 22000
        processed_data = vocoder(segment, f0, w, a, b, lowpassf, d, dB, chunk)
        segments.append(processed_data)
    final_output = np.concatenate(segments)
    return final_output, sr



# Execution code
if __name__ == "__main__":

    # Input Files--------
    video_file_path = "./VOICE.mp4"
    #backing_file_path = "./INST.wav"
    #midi_file_path = "./midi.mid"
    backing_file_path = "./metadata/music/3_exciting/exc_120_Db_0002.wav"
    midi_file_path = "./metadata/music/3_exciting/exc_120_Db_0002.mid"
    # Input Files--------

    audio_file_path = "./VOICE.wav"
    folder_path = './'
    first_audio_output_filename = "first_audio.wav"

    my_clip = mpe.VideoFileClip(video_file_path)
    my_clip.audio.write_audiofile(audio_file_path)
    my_clip.close()

    analyze_and_save_voice_file(audio_file_path, video_file_path, folder_path, backing_file_path, first_audio_output_filename)


    first_audio_file_path = folder_path + first_audio_output_filename
    vocoder_integrated_filename = folder_path + 'vocoder_integrated.wav'

    vocoder_audio_output, sr = simple_segment_audio_with_vocoder_integration(midi_file_path, first_audio_file_path, lowpassf, d, dB, chunk)
    sf.write(vocoder_integrated_filename, vocoder_audio_output, sr)

# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
#@title Clone repository
# Use git to clone the repository: git clone https://github.com/SociallyIneptWeeb/AICoverGen.git
# %cd AICoverGen

#@title Install requirements
# Install required packages using: pip install -r requirements.txt
# Update your system packages if needed.
# Install Sox using your package manager or download from its official website.

#@title Download MDXNet Vocal Separation and Hubert Base Models
# Run the model download script: python src/download_models.py

#@title Model Download Function

BASE_DIR = os.getcwd()

rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')

def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)

            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        raise Exception(f'No .pth model file was found in the extracted zip. Please check {extraction_folder}.')

    # move model and index file to extraction folder
    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    # remove any unnecessary nested folders
    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))

def download_online_model(url, dir_name):
    try:
        print(f'[~] Downloading voice model with name {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise Exception(f'Voice model directory {dir_name} already exists! Choose a different name for your voice model.')

        if 'pixeldrain.com' in url:
            url = f'https://pixeldrain.com/api/file/{zip_name}'

        urllib.request.urlretrieve(url, zip_name)

        print('[~] Extracting zip...')
        extract_zip(extraction_folder, zip_name)
        print(f'[+] {dir_name} Model successfully downloaded!')

    except Exception as e:
        raise Exception(str(e))

url = "-"

SONG_INPUT = "./vocoder_integrated.wav"
RVC_DIRNAME = "mal_nctmark"
PITCH_CHANGE = 0
PITCH_CHANGE_ALL = 4

INDEX_RATE = 0.5
FILTER_RADIUS = 3
PITCH_DETECTION_ALGO = "rmvpe"
CREPE_HOP_LENGTH = 128
PROTECT = 0.33
REMIX_MIX_RATE = 0.25
MAIN_VOL = 0
BACKUP_VOL = 0
INST_VOL = 0
REVERB_SIZE = 0.15

REVERB_WETNESS = 0.2
REVERB_DRYNESS = 0.8
REVERB_DAMPING = 0.7

OUTPUT_FORMAT = "mp3" 
import subprocess

command = [
    "python3",
    "./src/main.py",
    "-i", SONG_INPUT,
    "-dir", RVC_DIRNAME,
    "-p", str(PITCH_CHANGE),
    "-k",
    "-ir", str(INDEX_RATE),
    "-fr", str(FILTER_RADIUS),
    "-rms", str(REMIX_MIX_RATE),
    "-palgo", PITCH_DETECTION_ALGO,
    "-hop", str(CREPE_HOP_LENGTH),
    "-pro", str(PROTECT),
    "-mv", str(MAIN_VOL),
    "-bv", str(BACKUP_VOL),
    "-iv", str(INST_VOL),
    "-pall", str(PITCH_CHANGE_ALL),
    "-rsize", str(REVERB_SIZE),
    "-rwet", str(REVERB_WETNESS),
    "-rdry", str(REVERB_DRYNESS),
    "-rdamp", str(REVERB_DAMPING),
    "-oformat", OUTPUT_FORMAT
]


# Open a subprocess and capture its output

process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

# Print the output in real-time
for line in process.stdout:
    print(line, end='')

# Wait for the process to finish
process.wait()

#최종 파일 합치기

from moviepy.editor import *

# 파일 경로 설정
#audio_path1 = "./INST.wav"
audio_path2 = "./finale_out.wav"
video_path = "./first_video.mp4"

# 파일 로드
audio1 = AudioFileClip(backing_file_path)
audio2 = AudioFileClip(audio_path2)
video = VideoFileClip(video_path)

# 가장 짧은 길이 찾기
min_duration = min(audio1.duration, audio2.duration, video.duration)

# 파일 길이 잘라내기
audio1 = audio1.subclip(0, min_duration)
audio2 = audio2.subclip(0, min_duration)
video = video.subclip(0, min_duration)

# 오디오 합치기 (stereo channel이라고 가정)
if audio1.nchannels == 1:
    audio1 = audio1.to_stereo()
if audio2.nchannels == 1:
    audio2 = audio2.to_stereo()

mixed_audio = CompositeAudioClip([audio1.volumex(0.1), audio2.volumex(0.9)])

# 비디오와 오디오 합치기
final_clip = video.set_audio(mixed_audio)

# 결과 저장
final_clip.write_videofile("./output.mp4", codec="libx264", audio_codec="aac")