from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm


def make_prediction(args):

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(-1, 1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(step, 1), dtype=np.float32)
                tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch, dtype=np.float32)
        y_pred = model.predict(X_batch, verbose=0)
        y_mean = np.mean(y_pred, axis=0)
        y_pred_class = np.argmax(y_mean)
        real_class = os.path.dirname(wav_fn).split('/')[-1]
        print('Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred_class]))
        results.append(y_mean)

    np.save(os.path.join('logs', args.pred_fn), np.array(results))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default=None,
                        help='model file to make predictions (if None, predict all models)')
    parser.add_argument('--pred_fn', type=str, default=None,
                        help='fn to write predictions in logs dir (if None, use default names)')
    parser.add_argument('--src_dir', type=str, default='wavfiles',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    parser.add_argument('--all_models', type=bool, default=True,
                        help='predict all models if True')
    args, _ = parser.parse_known_args()

    # If all_models is True, predict for all three models
    if args.all_models or (args.model_fn is None and args.pred_fn is None):
        print("="*70)
        print("Predicting for all models...")
        print("="*70)
        
        models = [
            ('models/lstm.h5', 'y_pred_lstm'),
            ('models/conv1d.h5', 'y_pred_conv1d'),
            ('models/conv2d.h5', 'y_pred_conv2d')
        ]
        
        for model_path, pred_name in models:
            if os.path.exists(model_path):
                print(f"\n[{'='*68}]")
                print(f"Predicting with {model_path}...")
                print(f"[{'='*68}]")
                args.model_fn = model_path
                args.pred_fn = pred_name
                make_prediction(args)
                print(f"Saved predictions to logs/{pred_name}.npy\n")
            else:
                print(f"Warning: {model_path} not found. Skipping...\n")
    else:
        # Single model prediction
        if args.model_fn is None:
            args.model_fn = 'models/lstm.h5'
        if args.pred_fn is None:
            args.pred_fn = 'y_pred'
        
        make_prediction(args)

