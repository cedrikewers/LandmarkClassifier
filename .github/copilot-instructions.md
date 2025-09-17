# Landmark Classifier

A Flask-based web application that uses PyTorch deep learning models to classify landmark images. The application supports multiple pre-trained models (EfficientNet-B0, ResNet50, MobileNet V3) and provides web interface for image upload and classification with geolocation visualization.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap and Setup (Required for all deployments)
- Install dependencies: `pip3 install -r requirements.txt` -- takes 45 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
- Extract models and samples: `mkdir -p models label_samples && tar -xzf models.tar.gz -C models && tar -xzf label_samples.tar.gz -C label_samples` -- takes 10 seconds. NEVER CANCEL. Set timeout to 60+ seconds.

### Run the Application
- Development server: `python3 app.py` -- starts immediately, runs on http://127.0.0.1:5025
- Production server: `uwsgi uwsgi.ini` -- starts immediately, runs on http://127.0.0.1:5025

### Testing and Validation
- ALWAYS test the complete end-to-end workflow after making changes:
  1. Start the application using one of the methods above
  2. Navigate to http://127.0.0.1:5025 
  3. Upload an image from label_samples/ directory (e.g., label_samples/100028.jpg)
  4. Select a model (default: efficientnet-b0-v4)
  5. Click "classify" and verify results appear with probabilities and landmark names
  6. Verify geolocation map appears at bottom of results page
- Test image classification endpoint: `curl -X POST -F "image=@label_samples/100028.jpg" -F "model=efficientnet-b0-v4" http://127.0.0.1:5025/classify`
- Test main interface: `curl -s http://127.0.0.1:5025/`

### Docker Deployment (Known Limitation)
- Docker build: `docker build -t landmark-classifier .` -- may fail due to SSL certificate issues in sandbox environments
- If Docker build fails with SSL errors, use the Python/uwsgi deployment methods instead
- Docker run: `docker run -d -p 5025:5025 -e BASE_URL_PATH="/landmark" cedrikewers/landmark:latest` (if build succeeds)

## Project Structure and Key Files

### Core Application Files
- `app.py` -- Main Flask application with model loading and classification endpoints
- `requirements.txt` -- Python dependencies with specific PyTorch wheel URLs for CPU processing
- `uwsgi.ini` -- Production deployment configuration
- `Dockerfile` -- Container deployment configuration

### Models and Data
- `models.tar.gz` -- Pre-trained PyTorch models (650MB+) - must be extracted
- `label_samples.tar.gz` -- Sample images for testing (650MB+) - must be extracted
- `geolocations.json` -- Geographic data for landmark mapping
- `mapping.dict` -- Landmark ID to name mapping
- `manual_mapping.py` -- Manual landmark name overrides

### Templates
- `templates/index.jinja` -- Main upload interface
- `templates/results.jinja` -- Classification results display

### Available Models
The application supports these pre-trained models (configured in app.py lines 39-47):
- `efficientnet-b0-v3` -- EfficientNet-B0 model version 3
- `efficientnet-b0-v4` -- EfficientNet-B0 model version 4 (default)
- `ResNet50-v2` -- ResNet50 model version 2  
- `MobileNet_V3_Small-v2` -- MobileNet V3 Small model version 2

## Common Tasks and Troubleshooting

### Timing Expectations
- Dependency installation: ~45 seconds
- Model/sample extraction: ~10 seconds
- Application startup: Immediate (< 5 seconds)
- Image classification: 1-3 seconds per image
- NEVER CANCEL any of these operations - they complete quickly but may appear to hang initially

### Environment Requirements
- Python 3.12+ (tested with 3.12.3)
- Requires specific PyTorch/torchvision wheels for CPU processing (handled by requirements.txt)
- Linux x86_64 architecture (requirements.txt includes platform-specific wheels)

### Key Configuration
- Default port: 5025 (configurable in app.py and uwsgi.ini)
- Base URL path: configurable via `BASE_URL_PATH` environment variable
- Model selection: available models defined in `model_file_from_name` dict in app.py

### File Structure After Setup
```
/
├── .github/
│   └── copilot-instructions.md
├── models/                    # Extracted from models.tar.gz
│   ├── b0-10_000v3.pth
│   ├── b0-10_000v4.pth
│   ├── ResNet50-10_000v2.pth
│   └── MobileNet_V3_Small-v2.pth
├── label_samples/             # Extracted from label_samples.tar.gz
│   ├── 100028.jpg
│   ├── 100073.jpg
│   └── ...
├── templates/
│   ├── index.jinja
│   └── results.jinja
├── app.py
├── requirements.txt
├── uwsgi.ini
├── Dockerfile
└── ...
```

### Error Handling
- If port 5025 is in use: `ps aux | grep -E "(python|uwsgi)" | grep -v grep` to find running processes, then kill them
- If models fail to load: Verify models/ and label_samples/ directories exist and contain extracted files
- If SSL errors during pip install: The specific wheel URLs in requirements.txt should resolve this
- If Docker build fails: Use Python/uwsgi deployment instead - this is a known limitation in some environments

### Development Guidelines
- Always test with actual image uploads, not just API endpoints
- Verify the geolocation map renders correctly in results
- Test multiple model selections to ensure all work correctly
- Check that landmark name mapping works (some landmarks may show "None")
- Ensure the application can handle both development (`python3 app.py`) and production (`uwsgi`) deployment methods

## No CI/CD or Testing Infrastructure
This repository has no automated testing, linting, or CI/CD pipelines. All validation must be done manually using the testing procedures outlined above.