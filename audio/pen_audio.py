import numpy as np
import sounddevice as sd # type: ignore
import threading
import time
import wave
from pydub import AudioSegment
import os
from datetime import datetime

class AudioRecorder:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data = []
        self.lock = threading.Lock()
        
    def start_recording(self):
        """Start recording audio"""
        with self.lock:
            self.recording = True
            self.audio_data = []
            print("Recording started...")
            
    def stop_recording(self):
        """Stop recording audio"""
        with self.lock:
            self.recording = False
            print("Recording stopped...")
            
    def is_recording(self):
        """Check if currently recording"""
        return self.recording
            
    def add_audio_data(self, data):
        """Add audio data to the recording buffer"""
        if self.recording:
            with self.lock:
                # Convert to mono if needed and store as copy
                if len(data.shape) > 1:
                    audio_mono = data[:, 0].copy()
                else:
                    audio_mono = data.copy()
                self.audio_data.append(audio_mono)
                
    def save_to_mp3(self, filename=None):
        """Save recorded audio to MP3 file"""
        if not self.audio_data:
            print("No audio data to save!")
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"piano_recording_{timestamp}.mp3"
            
        try:
            # Concatenate all audio data
            with self.lock:
                audio_array = np.concatenate(self.audio_data, axis=0)
            
            # Normalize audio to prevent clipping
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val * 0.8  # Leave some headroom
            
            # Convert to 16-bit integers
            audio_16bit = (audio_array * 32767).astype(np.int16)
            
            # Save as WAV first (temporary file)
            wav_filename = filename.replace('.mp3', '.wav')
            with wave.open(wav_filename, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_16bit.tobytes())
            
            # Convert WAV to MP3 using pydub
            audio_segment = AudioSegment.from_wav(wav_filename)
            audio_segment.export(filename, format="mp3", bitrate="192k")
            
            # Clean up temporary WAV file
            os.remove(wav_filename)
            
            print(f"Audio saved as: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None

class AudioMixer:
    def __init__(self, sample_rate=44100, block_size=1024):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.active_notes = {}
        self.current_octave = 4  # Default octave
        self.lock = threading.Lock()
        self.running = True
        
        # Start the continuous audio stream
        self.stream = sd.OutputStream(
            callback=self.audio_callback,
            samplerate=sample_rate,
            channels=1,
            blocksize=block_size
        )
        self.stream.start()
        
    def generate_note_harmonics(self, freq, t, phase=0):
        """Generate harmonics"""
        wave = (0.4 * np.sin(2 * np.pi * freq * t + phase) +
                0.2 * np.sin(4 * np.pi * freq * t + phase) +
                0.1 * np.sin(6 * np.pi * freq * t + phase) +
                0.05 * np.sin(8 * np.pi * freq * t + phase))
        return wave
        
    def audio_callback(self, outdata, frames, time, status):
        """Called by sounddevice to generate audio in real-time"""
        with self.lock:
            if not self.active_notes:
                outdata.fill(0)
                return
                
            # Time array for this block
            t = np.arange(frames) / self.sample_rate
            mixed_audio = np.zeros(frames)
            
            # Generate audio for each active note
            for note_name, note_info in list(self.active_notes.items()):
                freq = note_info['frequency']
                phase = note_info['phase']
                start_time = note_info['start_time']
                
                # Generate the waveform
                wave = self.generate_note_harmonics(freq, t, phase)
                
                # Apply envelope (fade in at start, sustain while held)
                current_time = time.outputBufferDacTime
                note_age = current_time - start_time
                
                if note_age < 0.05:
                    envelope = note_age / 0.05
                else:
                    envelope = 1.0
                    
                wave *= envelope * 0.3  # Volume control
                mixed_audio += wave
                
                # Update phase for continuity between callbacks
                note_info['phase'] += 2 * np.pi * freq * frames / self.sample_rate
                
            # Normalize to prevent clipping when multiple notes play
            if len(self.active_notes) > 1:
                mixed_audio /= np.sqrt(len(self.active_notes))
                
            # Ensure we don't clip
            mixed_audio = np.clip(mixed_audio, -1.0, 1.0)
            outdata[:, 0] = mixed_audio
            
    def set_octave(self, octave):
        """Set the current octave (1-8)"""
        self.current_octave = max(1, min(8, octave))

    def start_note(self, note_name, base_frequency):
        """Start playing a continuous note with octave adjustment"""
        with self.lock:
            if note_name not in self.active_notes:
                # P_FREQ has base octive of one
                frequency = base_frequency * (2 ** (self.current_octave - 1))
                self.active_notes[note_name] = {
                    'frequency': frequency,
                    'phase': 0,
                    'start_time': time.time()
                }
                
    def stop_note(self, note_name):
        """Stop playing a note"""
        with self.lock:
            if note_name in self.active_notes:
                del self.active_notes[note_name]
                
    def update_notes(self, detected_keys):
        """
        Update which notes are playing based on detected keys
        detected_keys: set of note names currently being touched
        """
        if not hasattr(self, 'previous_keys'):
            self.previous_keys = set()
            
        current_keys = set(detected_keys)
        
        # Start new notes
        new_keys = current_keys - self.previous_keys
        for key in new_keys:
            if key in P_FREQ:
                self.start_note(key, P_FREQ[key])
                
        # Stop released notes
        released_keys = self.previous_keys - current_keys
        for key in released_keys:
            self.stop_note(key)
            
        self.previous_keys = current_keys
        
    def cleanup(self):
        """Clean up audio resources"""
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            
    def get_active_notes(self):
        """Get list of currently playing notes"""
        with self.lock:
            return list(self.active_notes.keys())

class RecordableAudioMixer(AudioMixer):
    def __init__(self, sample_rate=44100, block_size=1024):
        self.recorder = None
        super().__init__(sample_rate, block_size)
        
    def set_recorder(self, recorder):
        """Set the audio recorder"""
        self.recorder = recorder
        
    def audio_callback(self, outdata, frames, time, status):
        """Enhanced audio callback that also records"""
        # Call the original audio generation
        super().audio_callback(outdata, frames, time, status)
        
        # If recorder is set and recording, capture the audio
        if self.recorder and self.recorder.is_recording():
            self.recorder.add_audio_data(outdata)

# Piano frequencies
freq_multiplier = 16
P_FREQ = {
    "C": 16.35, "C#": 17.32, "Db": 17.32,
    "D": 18.35, "D#": 19.45, "Eb": 19.45, "E": 20.6, "F": 21.83,
    "F#": 23.12, "Gb": 23.12, "G": 24.5, "G#": 25.96, "Ab": 25.96,
    "A": 27.5, "A#": 29.14, "Bb": 29.14, "B": 30.87
}