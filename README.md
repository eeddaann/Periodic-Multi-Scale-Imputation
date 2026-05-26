# Periodic-Multi-Scale-Imputation

## Repository Structure

```
├── pmsi/                      # Core package engine
│   ├── evaluation.py          # Functions for evaluation
│   ├── imputers.py            # Baseline models implementations
│   ├── pmsi_core.py           # PMSIImputer core framework logic & optimization
│   └── visualization.py       # Script to generate the figures
├── requirements.txt           # Requirements for installation
├── reproduce_results.ipynb    # Jupyter notebook to demonstrate the framework
```
Installation
Clone the Repository:

```Bash
git clone https://github.com/eeddaann/Periodic-Multi-Scale-Imputation.git
cd Periodic-Multi-Scale-Imputation

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```