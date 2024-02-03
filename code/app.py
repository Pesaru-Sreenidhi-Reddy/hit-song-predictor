from flask import Flask, request, jsonify, render_template

import scipy.signal as signal
import joblib
import numpy as np
import sklearn
import pandas as pd

import librosa
import requests



app = Flask(__name__)

#Loading the XGBoost Model 

xgbmodel = joblib.load('spotifyXGB')


def calculate_valence(audio_file):
    # Load audio
    y, sr = librosa.load(audio_file)

    # Extract spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
    
    # Extract timbral features
    rms = librosa.feature.rms(y=y)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]
    
    # Extract rhythm features
    tempogram = librosa.feature.tempogram(y=y, sr=sr)[0]

    # Combine features (spectral, timbral, and rhythm)
    all_features = np.vstack([
        spectral_centroid, spectral_bandwidth, spectral_contrast,
        rms, zero_crossing_rate, tempogram
    ])

    # Calculate some statistics or summary from these features
    feature_mean = np.mean(all_features, axis=1)
    valence= np.sum(feature_mean)  # Example calculation

    return valence   


    
@app.route("/", methods=['GET'])
def Home():
    return render_template('audio.html')



@app.route('/upload_audio', methods=['POST'])
def upload_audio():
        if request.method == 'POST':
            file = request.files['file']
            #calculation of features
            if file:
                audio_file_path =   file.filename 
                file.save(audio_file_path)
                y, sr = librosa.load(audio_file_path)
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                # Extracting beat-related features
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                beat_strength = np.mean(librosa.feature.rms(y=y))
                tempo_stability = np.std(np.diff(librosa.frames_to_time(beats, sr=sr)))
                beat_times = librosa.frames_to_time(beats, sr=sr)
                chorus_hit = beat_times[0] if len(beat_times) > 0 else None
                sections = len(librosa.effects.split(y))
                energy_rms = librosa.feature.rms(y=y)
                energy = np.mean(energy_rms)
                loudness = np.mean(y)
                

                # Calculating danceability based on combined features
                danceability = (beat_strength * tempo) / tempo_stability
                danceability= danceability / 2035

                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                # Calculate the chromagram
                chroma = np.abs(librosa.feature.chroma_stft(y=y, sr=sr))

                # Calculate the mean along the time axis to get a single chroma vector
                mean_chroma = np.mean(chroma, axis=1)

                # Define a mapping of pitch classes
                pitch_classes = [-1,0,1 , 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

                # Find the index of the maximum value in the mean chroma vector
                max_index = np.argmax(mean_chroma)

                # Determine the key from the maximum index
                key = pitch_classes[max_index]
               

                # Determine the mode (major or minor) based on the pattern of intervals
                major_intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
                minor_intervals = [0, 2, 3, 5, 7, 8, 10]  # Minor scale intervals

                # Calculate the difference between adjacent elements to find intervals
                diff = np.diff(mean_chroma)
                periodicity = np.abs(signal.find_peaks(diff)[0])

                # Determine whether it aligns better with major or minor scale intervals
                major_score = np.sum(np.isin(periodicity, major_intervals))
                minor_score = np.sum(np.isin(periodicity, minor_intervals))

                # Choose the mode based on the higher score
                mode = 1 if major_score > minor_score else 0

                duration_ms = librosa.get_duration(y=y, sr=sr) * 1000
                time_signature=4#most common
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

                # Perform simple thresholding to identify speech segments based on MFCCs
                threshold = 500  # Adjust the threshold as needed
                speech_segments = np.mean(mfccs, axis=0) > threshold

                # Calculate speechiness ratio
                total_frames = len(speech_segments)
                speech_frames = np.sum(speech_segments)
                speechiness = speech_frames / total_frames
                acousticness_spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                acousticness = np.mean(acousticness_spectral_centroid)

                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

                # Calculate ratio of spectral centroid to spectral bandwidth
                ratio = acousticness / spectral_bandwidth

                # Calculate instrumentalness based on the ratio
                # Example: If ratio is below a certain threshold, consider it instrumental
                threshold = 0.5  
                instrumentalness = sum(ratio < threshold) / len(ratio)
                valence = calculate_valence(audio_file_path)
                
                valence =  valence/6053
                rms = librosa.feature.rms(y=y)[0]
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]
                liveness = (
                            acousticness +
                            np.sum(rms) +
                            np.sum(zero_crossing_rate) +
                            tempo
                            )
                liveness=liveness/12557
                acousticness = acousticness/38000
                
                input_variables = pd.DataFrame([[danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature,chorus_hit,sections]])
                input_variables.columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'chorus_hit', 'sections']
                
                pd.set_option('display.max_columns', None)
                print(input_variables)
                prediction = xgbmodel.predict(input_variables)
        
                output = int(prediction[0])
                if output==1:
                    print("Congratulations, your song is hit")
                    return render_template('audio.html',prediction_texts="Congratulations, your song has a high chance of making it onto the Billboard Hot 100 list.")
                else:
                   print("Sorry, your song has less chance to get onto billBoard Hot 100")
                   return render_template('audio.html', prediction_texts="Sorry, chance of getting this song on Billboard Hot 100 list is low")
        else:
            return render_template('audio.html',prediction_texts="Something went wrong!")



if __name__ == '__main__':
    app.run(debug=True)
