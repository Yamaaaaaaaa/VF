import librosa
import numpy as np
import io

TARGET_SR = 16_000   # Hz
TARGET_DURATION = 5.0      # giây
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SC_N_BANDS = 6
SC_FMIN = 200.0

def process_audio_file(file_bytes: bytes) -> np.ndarray:
    """
    Process an uploaded audio file from bytes.
    Extract exactly the 99D feature vector required for pgvector cosine search.
    """
    # Load and resample using librosa (requires PySoundFile or Audioread in background)
    # Wrap bytes in BytesIO so soundfile/librosa can read it
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=TARGET_SR, mono=True)
    
    # Pad or clip to exactly 5 seconds
    if len(y) >= TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]
    else:
        pad_len = TARGET_SAMPLES - len(y)
        y = np.pad(y, (0, pad_len), mode='constant')

    # Extract features
    # 1. MFCC mean (40) and std (40)
    mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=N_MFCC,
                                n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
    mfcc_std  = np.std(mfcc, axis=1).astype(np.float32)

    # 2. Spectral Contrast mean (7) - as SC_N_BANDS=6 gives 7 bands
    sc = librosa.feature.spectral_contrast(y=y, sr=TARGET_SR,
                                           n_bands=SC_N_BANDS, fmin=SC_FMIN)
    sc_mean = np.mean(sc, axis=1).astype(np.float32)

    # 3. Chroma STFT mean (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=TARGET_SR,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
    chroma_mean = np.mean(chroma, axis=1).astype(np.float32)

    # Concatenate to 99-dimensional vector
    embedding = np.concatenate([mfcc_mean, mfcc_std, sc_mean, chroma_mean])
    
    return embedding


def _analyze_signal(y: np.ndarray, sr: int) -> dict:
    """Core analysis: waveform, mel spectrogram, MFCC matrix, 99D embedding."""
    if len(y) >= TARGET_SAMPLES:
        y = y[:TARGET_SAMPLES]
    else:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode='constant')

    # ① Waveform — downsample to 600 pts
    step = max(1, len(y) // 600)
    waveform = y[::step][:600].tolist()

    # ② Mel Spectrogram (64 bands)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32).tolist()

    # ③ MFCC matrix (40 × frames)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mfcc_matrix = mfcc.astype(np.float32).tolist()

    # ④ 99D embedding (recompute from signal)
    mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
    mfcc_std  = np.std(mfcc,  axis=1).astype(np.float32)
    sc = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=SC_N_BANDS, fmin=SC_FMIN)
    sc_mean = np.mean(sc, axis=1).astype(np.float32)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    chroma_mean = np.mean(chroma, axis=1).astype(np.float32)
    embedding = np.concatenate([mfcc_mean, mfcc_std, sc_mean, chroma_mean]).tolist()

    return {
        "waveform": waveform,
        "mel_spectrogram": mel_db,
        "mfcc_matrix": mfcc_matrix,
        "embedding": embedding,
    }


def analyze_from_path(file_path: str) -> dict:
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    return _analyze_signal(y, sr)
