# Claude Instructions for Gene Circuit Design Project

## Project Overview
This is a gene circuit design project focused on active learning for DNA sequence-expression prediction. The project includes machine learning models for predicting gene expression from DNA sequences.

## Key Files and Structure
- `run_experiments.py`: Main experiment runner for active learning
- `main.py`: Simple entry point
- `utils/`: Core utilities including metrics and sequence processing
- `data/`: Contains 384-well plate data and embeddings
- `results/`: Various experimental results with different strategies
- `notebooks/`: Jupyter notebooks for visualization
- `test/`: Test files

## Development Commands
- **Linting**: `ruff check .` or `ruff check --fix .`
- **Testing**: `pytest` (runs tests in test/ directory)
- **Main execution**: `python run_experiments.py`

## Code Style
- Uses Ruff for linting and formatting
- Line length: 88 characters
- Python 3.8+ compatibility required
- Follow existing patterns in utils/ modules

## Dependencies
- Core: numpy, pandas, scikit-learn, scipy, tqdm
- Dev: pytest, ruff, torch, matplotlib, seaborn
- Uses uv for dependency management (uv.lock present)

## Active Learning Context
The project implements various active learning strategies:
- Random sampling
- High expression targeting
- Log likelihood-based selection
- Embedding-based approaches

## Important Notes
- Multiple experimental result directories exist with different configurations
- Sequence data includes both raw sequences and embeddings
- Results include custom metrics and cross-validation approaches
- Git status shows modifications to run_experiments.py and utils/sequence_utils.py

## Testing
- Test files are in `test/` directory
- Use `pytest` to run all tests
- Markers available: `slow` for time-intensive tests

## Data Files
- 384-well plate data with plasmid maps and sequences
- RICE embeddings for sequence representation
- Expression data mapping sequences to measured outputs
