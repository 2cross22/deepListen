# deepListen - Advanced Music Genre Classification

A sophisticated music genre classification system powered by the MUSAIC (MUsic SAmple Identification Classifier) model, achieving 90% accuracy across 10 major music genres.

## About

deepListen is an AI-powered music genre classification platform that leverages deep learning to accurately identify and categorize music into distinct genres. The system uses the proprietary MUSAIC model, which stands for "MUsic SAmple Identification Classifier," designed specifically for high-precision audio analysis.

## Features

- **High Accuracy**: 90% classification accuracy across 10 major music genres
- **Multiple Format Support**: Handles MP3, WAV, M4A, and FLAC audio files
- **Real-time Analysis**: Instant genre prediction with confidence scores
- **Web Interface**: Clean, modern web application for easy file upload and analysis
- **Detailed Results**: Comprehensive genre probability breakdown for each prediction
- **Secure Processing**: Files are processed securely and deleted after analysis

## Supported Genres

The MUSAIC model can classify music into the following 10 genres:
- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## Technology Stack

- **Backend**: Flask (Python)
- **AI Model**: MUSAIC (Custom-trained neural network)
- **Audio Processing**: Librosa, TensorFlow
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Audio Features**: YAMNet embeddings for robust feature extraction

## Model Architecture

The MUSAIC model utilizes a sophisticated pipeline:

1. **Audio Preprocessing**: Audio files are loaded and resampled to 16kHz
2. **Feature Extraction**: YAMNet embeddings are extracted to capture rich audio features
3. **Dimensionality Reduction**: Features are averaged to create compact representations
4. **Classification**: A custom-trained neural network performs genre classification
5. **Confidence Scoring**: Probability distributions are calculated for all genres

## Accuracy Performance

The MUSAIC model achieves:
- **Overall Accuracy**: 90%
- **Cross-genre Performance**: Consistent performance across all 10 genres
- **Confidence Scoring**: Reliable confidence metrics for predictions

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepListen.git
cd deepListen
```

2. Install dependencies:
```bash
pip install -e .
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Upload Audio**: Drag and drop or click to upload your audio file
2. **Analysis**: The MUSAIC model processes your file and extracts features
3. **Results**: View the predicted genre with confidence scores and probability breakdown
4. **Download**: Print or save your results for future reference

## API Endpoints

- `GET /`: Main application interface
- `POST /upload`: File upload and genre classification
- `POST /api/predict`: REST API for programmatic access
- `GET /health`: Health check endpoint

## Development

### Training the Model
To retrain the MUSAIC model with new data:
```bash
python MUSAICtrainer.py
```

### Testing
Run the test suite:
```bash
python tester.py
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). This means:

- You are free to share and adapt the material
- You must provide appropriate attribution
- You may not use this material for commercial purposes

For full license details, see the [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- YAMNet for audio feature extraction
- TensorFlow and Keras for deep learning framework
- Librosa for audio processing capabilities
- The open-source community for various supporting libraries

## Contact

For questions, support, or collaboration opportunities, please open an issue on GitHub.

---

**deepListen** - Powered by MUSAIC AI Technology 