import numpy as np
import sounddevice as sd # type: ignore
import threading
import time

class AudioMixer:
    def __init__(self, sample_rate=44100, block_size=1024):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.active_notes = {} 
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
            
    def start_note(self, note_name, frequency):
        """Start playing a continuous note"""
        with self.lock:
            if note_name not in self.active_notes:
                self.active_notes[note_name] = {
                    'frequency': frequency,
                    'phase': 0,
                    'start_time': time.time()
                }
                print(f"Started note: {note_name}")
                
    def stop_note(self, note_name):
        """Stop playing a note"""
        with self.lock:
            if note_name in self.active_notes:
                del self.active_notes[note_name]
                print(f"Stopped note: {note_name}")
                
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

# Piano frequencies
P_FREQ = {
    "C3": 130.81, "C#3": 138.59, "Db3": 138.59, "D3": 146.83, "D#3": 155.56,
    "Eb3": 155.56, "E3": 164.81, "F3": 174.61, "F#3": 184.99, "Gb3": 184.99,
    "G3": 195.99, "G#3": 207.65, "Ab3": 207.65, "A3": 220.00, "A#3": 233.08,
    "Bb3": 233.08, "B3": 246.94, "C4": 261.63, "C#4": 277.18, "Db4": 277.18,
    "D4": 293.66, "D#4": 311.13, "Eb4": 311.13, "E4": 329.63, "F4": 349.23,
    "F#4": 369.99, "Gb4": 369.99, "G4": 392.00, "G#4": 415.30, "Ab4": 415.30,
    "A4": 440.00, "A#4": 466.16, "Bb4": 466.16, "B4": 493.88, "C5": 523.25,
    "C#5": 554.37, "Db5": 554.37, "D5": 587.33, "D#5": 622.25, "Eb5": 622.25,
    "E5": 659.25, "F5": 698.46, "F#5": 739.99, "Gb5": 739.99, "G5": 783.99,
    "G#5": 830.61, "Ab5": 830.61, "A5": 880.00, "A#5": 932.33, "Bb5": 932.33,
    "B5": 987.77, "C6": 1046.50
}


"""

mixer = AudioMixer()

try:
    while True:
        ret, frame = cap.read()
        
        # detected_keys = detect_finger_on_keys(frame)  # Returns list like ['C4', 'E4']
        
        # mixer.update_notes(detected_keys)
        
        cv2.imshow('Piano', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    mixer.cleanup()
    cap.release()
    cv2.destroyAllWindows()
"""