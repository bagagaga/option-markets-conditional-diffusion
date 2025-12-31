# Option Markets Conditional Diffusion

Conditional DDPM for option price prediction with MLOps pipeline using PyTorch Lightning.

## Overview

**Framework**: PyTorch Lightning 2.6+  
**Model**: Conditional Denoising Diffusion Probabilistic Model (DDPM)  
**Input**: Market features (prices, Greeks, returns, temporal)  
**Output**: Probabilistic predictions for option price changes  
**Dataset**: S&P 500 options panel (~21,562 observations)  
**Data Versioning**: DVC with S3 remote storage

## Quick Start

### Method 1: Docker (Recommended)

```bash
# Start services
docker-compose -f docker-compose.prod.yml up -d

# Run training
docker exec option-diffusion-trainer bash -c "cd /workspace && python -m option_diffusion.train"

# Or use convenience script
docker exec option-diffusion-trainer bash -c "./scripts/train.sh"

# Run inference
docker exec option-diffusion-trainer bash -c "./scripts/test.sh"
```

**Services**:
- MLflow UI: http://localhost:5051
- PostgreSQL: localhost:5433
- Trainer: Interactive container with mounted workspace

### Method 2: Local

```bash
poetry install
poetry run python -m option_diffusion.train
```

## Training

### Basic Training
```bash
# Docker
docker exec option-diffusion-trainer bash -c "cd /workspace && python -m option_diffusion.train"

# Local
poetry run python -m option_diffusion.train
```

### Custom Parameters
```bash
# Change epochs and batch size
python -m option_diffusion.train training.epochs=100 training.batch_size=256

# Modify architecture
python -m option_diffusion.train model.architecture.hidden_dim=128

# Adjust learning rate
python -m option_diffusion.train training.optimizer.lr=0.0001
```

### Expected Output
```
Training for 100 epochs...
Epoch 001/100: train_loss=0.8838, val_loss=0.7355
Epoch 010/100: train_loss=0.6791, val_loss=0.6562
...
Training completed successfully!
Best validation loss: 0.6130
```

## Inference

```bash
# Docker
docker exec option-diffusion-trainer bash -c "cd /workspace && python -m option_diffusion.test"

# Local
poetry run python -m option_diffusion.test
```

Generates predictions and saves to `results/predictions.csv`.

## Configuration

Edit `configs/config.yaml` to customize:
- Model architecture: `model.architecture.hidden_dim`, `model.diffusion.timesteps`
- Training: `training.epochs`, `training.batch_size`, `training.optimizer.lr`
- Data source: `data.url` (public URL) or S3 credentials

Override at runtime:
```bash
python -m option_diffusion.train training.epochs=10 training.batch_size=64
```

### S3 Configuration (Optional)

By default, the project uses a public dataset URL. To use private S3 storage:

**Option 1: Environment Variables**
```bash
export YC_S3_ENDPOINT="https://storage.yandexcloud.net"
export YC_S3_ACCESS_KEY="your_access_key"
export YC_S3_SECRET_KEY="your_secret_key"
export YC_S3_BUCKET="diploma-finance-data"
export YC_S3_REGION="ru-central1"
```

**Option 2: Runtime Override**
```bash
python -m option_diffusion.train \
  data.url=null \
  s3.access_key=YOUR_KEY \
  s3.secret_key=YOUR_SECRET \
  s3.bucket=YOUR_BUCKET
```

**Option 3: Docker Compose**

Add to `docker-compose.prod.yml` under `trainer.environment`:
```yaml
- YC_S3_ACCESS_KEY=your_key
- YC_S3_SECRET_KEY=your_secret
- YC_S3_BUCKET=diploma-finance-data
```

## Project Structure

```
option_diffusion/
├── train.py          # Training pipeline
├── test.py           # Inference
├── model.py          # DDPM architecture
├── data_utils.py     # Data loading
├── hedging.py        # P&L analysis
└── constants.py      # Global constants

configs/
├── config.yaml       # Main config
├── model/            # Model configs
├── data/             # Data configs
└── training/         # Training configs

docker-compose.prod.yml  # Production services
Dockerfile.mlflow        # Custom MLflow image
```

## Model Details

**Architecture**: TinyCondEpsNet (MLP with 2,881 parameters)  
**Diffusion**: 100 timesteps, linear beta schedule (1e-4 to 2e-2)  
**Optimizer**: Adam (lr=1e-3, weight_decay=0.0)  
**Features**: 10 market indicators (C_close, F_close, tau_yrs, m, C_ret1, F_ret1, C_d1, F_d1, dow_sin, dow_cos)  
**Target**: `dC_next` (next-period option price change)

### Performance Metrics (3 epochs)
- Train Loss: 0.8838 → 0.6791
- Val Loss: 0.7355 → 0.6562
- Test MSE: 773.06
- Test MAE: 19.21

## Data Pipeline

**Default**: Public URL (no credentials required)
```
https://storage.yandexcloud.net/option-dataset/spx_multiline_panel.csv
```

**Features**:
- Automatic data loading from public URL
- Feature standardization (mean=0, std=1)
- Target normalization
- Time-based train/val split (80/20, no data leakage)
- Normalization stats saved in checkpoints

**Dataset Statistics**:
- Total samples: 21,562
- Training samples: 16,488 (36 dates)
- Validation samples: 5,074 (10 dates)
- Date range: 2025-10-20 to 2025-12-23

## Monitoring

PyTorch Lightning + MLflow tracks:
- All hyperparameters
- Training and validation losses per epoch
- Model checkpoints (best + last)
- Training curves via Lightning callbacks

Access MLflow UI at http://localhost:5051 to view:
- Experiment runs and comparisons
- Hyperparameter tuning results
- Training curves and metrics
- Model artifacts and checkpoints

## Convenience Scripts

Located in `scripts/` directory for use inside Docker:

```bash
# Training (with dependency installation)
docker exec option-diffusion-trainer bash -c "./scripts/train.sh"

# Training with custom parameters
docker exec option-diffusion-trainer bash -c "./scripts/train.sh training.epochs=50"

# Inference
docker exec option-diffusion-trainer bash -c "./scripts/test.sh"
```

## Development

### Code Quality

Pre-commit hooks are configured but need installation:

```bash
# Install pre-commit (host machine)
poetry install
poetry run pre-commit install

# Run manually on all files
poetry run pre-commit run --all-files
```

**Pre-commit checks**:
- Ruff linting and formatting
- Trailing whitespace removal
- YAML/JSON/TOML validation
- Large file detection
- Merge conflict detection
- Prettier formatting for configs

**Note**: Pre-commit runs automatically before git commits after installation.

### Testing

```bash
# Test training (3 epochs)
docker exec option-diffusion-trainer bash -c "cd /workspace && python -m option_diffusion.train training.epochs=3"

# Test inference
docker exec option-diffusion-trainer bash -c "cd /workspace && python -m option_diffusion.test"

# Verify MLflow connection
docker exec option-diffusion-trainer bash -c "curl http://mlflow-prod:5000/health"
```

## Docker Services

### Service Details

**mlflow-prod**:
- Image: Custom (Dockerfile.mlflow with psycopg2-binary)
- Port: 5051 → 5000
- Backend: PostgreSQL
- Purpose: Experiment tracking and model registry

**postgres-prod**:
- Image: postgres:16.3-alpine
- Port: 5433 → 5432
- Database: MLflow backend storage
- Volume: Persistent data storage

**trainer**:
- Image: Custom (Dockerfile with PyTorch + dependencies)
- Volumes: Workspace, data, models, logs mounted
- Working Dir: /workspace
- Command: Interactive (tail -f /dev/null)

### Container Management

```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Stop services
docker-compose -f docker-compose.prod.yml down

# Stop and remove volumes
docker-compose -f docker-compose.prod.yml down -v

# Rebuild services
docker-compose -f docker-compose.prod.yml up -d --build

# View logs
docker logs mlflow-prod
docker logs postgres-prod
docker logs option-diffusion-trainer

# Enter container
docker exec -it option-diffusion-trainer bash
```
