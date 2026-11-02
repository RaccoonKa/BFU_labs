# Fundamentals of Audio Processing

An educational course on digital audio signal processing and basic analysis of audio data. Built around hands-on practice with Python, NumPy, SciPy, and Matplotlib: from loading and visualizing a signal to frequency and spectral analysis.

## What the course is about

Modern audio processing relies on methods that turn continuous acoustic vibrations into digital sequences of samples and make it possible to extract information from them — from amplitude and frequency to spectral characteristics and temporal dynamics. The course walks through the key stages of working with such data: loading and preparing the signal, visualization, the Fourier transform, and spectral analysis.

**Data.** A digital audio signal is a sequence of samples taken at a certain sampling rate. The sampling rate (e.g., 44100 Hz) determines which frequencies the signal can represent, while bit depth and data type (float, int16) affect precision and dynamic range. The duration of a file and the number of samples are related by a simple formula: `duration = len(signal) / sr`. Multi-channel recordings are reduced to mono by averaging the channels.

**Amplitude-time waveform.** A plot of the signal over time shows how amplitude is distributed across the duration: attacks, decays, pauses, and the overall character of the sound. It is the primary, intuitive way to assess the structure of audio.

**Fourier transform.** The fast Fourier transform (FFT, in practice `rfft`) decomposes the signal into frequency components and shows which frequencies are present in the recording and with what amplitude. The frequency spectrum is useful for identifying tonal components, noise, and characteristic resonances.

**Spectrogram.** A spectrogram combines time and frequency: it shows how the spectrum of the signal changes over time. It is a key tool for analyzing the dynamics of sound — attacks, transitions, and rhythmic structures. A logarithmic representation in decibels (`10*log10`) makes weak components visible.

**Harmonic and percussive parts.** Any musical signal can be roughly divided into a harmonic part (stable periodic oscillations that form pitch and melody) and a percussive part (short impulsive components responsible for rhythm and attacks). Understanding this separation underlies many tasks — from source separation to rhythmic analysis.

## Course structure

| # | Topic | Key tools |
|---|-------|-----------|
| 1 | Signal loading, file parameters, amplitude-time waveform | `scipy.io.wavfile`, `numpy`, `matplotlib` |
| 2 | Frequency spectrum via the Fourier transform | `scipy.fft` (`rfft`, `rfftfreq`) |
| 3 | Spectrogram and time-frequency analysis | `scipy.signal.spectrogram` |
| 4 | Harmonic and percussive components of a signal (theory) | — |