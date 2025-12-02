import random
import librosa


class TransformCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):

        for transform in self.transforms:
            x = transform(x)

        return x

class RandomGain:
    
    def __init__(self, min_db=-12, max_db=6):
        self.min_db = min_db
        self.max_db = max_db

    def __call__(self, x):

        db = random.uniform(self.min_db, self.max_db)
        gain = 10 ** (db / 20.)
        x *= gain

        return x

class RandomPitch:
    
    def __init__(self, sr, min_pitch=-3, max_pitch=3):
        self.sr = sr
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

    def __call__(self, x):

        pitch = random.uniform(self.min_pitch, self.max_pitch)
        x = librosa.effects.pitch_shift(x, sr=self.sr, n_steps=pitch)

        return x

class RandomGainPitch:
    def __init__(self, sr, min_db=-12, max_db=6, min_pitch=-3, max_pitch=3):
        self.gain_transform = RandomGain(min_db=min_db, max_db=max_db)
        self.pitch_transform = RandomPitch(sr=sr, min_pitch=min_pitch, max_pitch=max_pitch)

    def __call__(self, x):
        x = self.gain_transform(x)
        x = self.pitch_transform(x)
        return x