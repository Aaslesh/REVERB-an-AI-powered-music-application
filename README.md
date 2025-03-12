# REVERB: An AI-Powered Music Application

REVERB is an end-to-end music analysis system implementing three core machine learning paradigms:
1. **MOD1**: Supervised Audio Classification
2. **MOD2**: Transfer Learning with Architectural Adaptation
3. **MOD3**: Content-Based Recommendation Systems

**Architecture Overview**:
- Pre-trained CNN with Mel-spectrogram feature extraction
- Sliding window analysis with 50% overlap (4s windows @ 22.05kHz)
- Majority voting ensemble over temporal chunks

**Audio Storage Schema**:
- 'genres_data/' has training data 
- 'classes/' has new genre proposals and cleared after post-migration
- 'songs/unclassified' Inference inputs and cleared post-processing
- 'spotify_millsongdata.csv' stores Lyric corpus
- 'recommend.h5' stores TF-IDF matrix
- 'model_V3.h5' stores cnn models layer and its weights

**Training Regimen**:
- 30 epochs with early stopping (patience=5)
- Batch normalization with 0.5 dropout
- Adam optimizer (β₁=0.9, β₂=0.999)

**Input Specifications**:
Supported formats: PCM-encoded WAV, MP3 (44.1kHz/16bit)
Input directory: 'songs/unclassified/' 'genres_data/'
Minimum viable duration: 4 seconds

**Key Dependencies**:
- Python==3.10.13
- librosa==0.9.1
- tensorflow==2.10.0
- scikit-learn==1.2.2

## Key Features

### MOD1: Intelligent Genre Classification
- Automatically organizes unclassified music files using a trained a model—either built from scratch or fine-tuned model(which accommodate new genres).
- Supports MP3 and WAV formats; newly classified files are added to the `genres_data` directory for further fine-tuning or for building a model from scratch.
- Performs chunk-based analysis with overlap handling.
- Extracts features from the mel-spectrogram of audio data.
- Provides silent error handling and progress reporting.


### MOD2: Adaptive Model Training
- **Continuous Learning:** When a new class (folder) is detected in the `classes` folder, the system extracts unique elements from that class. If the class already exists in `genres_data`, only the unique elements are added; otherwise, the entire folder is migrated to `genres_data`. This data is then used for fine-tuning or building a model from scratch.
- **Dynamic Architecture Adaptation:** If an existing model is detected, 80% of its layers will be frozen and 20% are fine-tuned using the available data by changing the architecture to add a new node to the model. Otherwise, a new model is built from scratch.
- Incorporates automatic model versioning.
- Provides comprehensive performance metrics.
- Supports GPU-accelerated training.


### MOD3: Content-Based Recommendations
- Fuzzy matching of song titles.
- TF-IDF analysis of lyrical content.
- Efficient data storage using HDF5.
- Command-line interface (CLI) for user interaction.
- Similarity scoring utilizing sparse matrices.

### Requirements
- Python 3.8+
- NVIDIA GPU (Recommended for MOD2)
- 8GB+ RAM

**References**
- McFee et al., "librosa: Audio and Music Signal Analysis in Python", 2015
- Abadi et al., "TensorFlow: Large-Scale Machine Learning", 2016
- Logan, "Mel Frequency Cepstral Coefficients for Music Modeling", 2000

### Setup
```bash
# Clone the repository 
git clone https://github.com/Aaslesh/REVERB-an-AI-powered-music-application.git

# Install dependencies
pip install -r requirements.txt

# Create directory structure if needed
mkdir -p genres_data/classical genres_data/rock  # Add other genre folders(which has audio samples) as needed
mkdir -p songs/unclassified songs/classified # Add songs to do unclassified folder as needed which then will be classified according to their genre in 'songs/classified/<genres>'

# Delete the 'model_V3.h5' and 'recommend.h5' if you want to build a model according to your database 

# Add the genres folders which has audio samples, to the 'genres_data' and add the song lyrical data to 'spotify_millsongdata.csv'

