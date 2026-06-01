## Intelligent Vehicle Dynamics Estimation & Control (Hybrid Transformer-EKF Architecture)

A data-driven, real-time cyber-physical system featuring a **Two-Stage Hybrid Architecture**. This system combines deterministic physics-based state estimators (**Extended Kalman Filters / Unscented Kalman Filters**) with deep sequence models (**Force Transformers**) to calculate transient vehicle states, dynamic tire forces, and virtual sensor metrics from noisy telemetry data.

The repository supports two major validation branches:
1. Real-world sensor data via the **TUM dataset**.
2. Converted simulated profiles via the **Vehicle Simulation dataset**.

---

## 🛠 System & Repository Architecture

This repository bridges classical control theory with modern deep learning. The file tree below represents the structural design of the project components:

```text
├── codes_allFilters_architectures/     # Main unified execution workflows
│   ├── nclt_env/                        # Python virtual environment
│   ├── train_newarch.py                 # Unified training entrypoint for Transformer
│   ├── run_newarch.py                   # Unified evaluation entrypoint for Hybrid model
│   ├── run_ukf.py                       # Baseline classical UKF runner
│   ├── tum_ukf.py                       # Underlying UKF physics configuration
│   └── convert_sim_to_tum_csv.py        # Mapping script for data unification
│
├── codes_2Stage_hybrid_architecture/   # Core algorithms & modules
│   ├── RUN_GUIDE_miniT.md               # Quick execution notes
│   ├── vehicle_force_ekf.py            # Physics-based Extended Kalman Filter
│   ├── sim_force_ekf.py                # Simulation verification wrapper for EKF
│   ├── sim_train_force_transformer.py  # PyTorch training pipeline for the Transformer model
│   ├── force_transformer_model.pt      # Trained weights for deep-learning virtual sensor
│   ├── force_transformer_common.py    # Transformer model architecture & definitions
│   ├── vehicle_filter_common.py        # Signal conditioning & preprocessing utilities
│   ├── prepare_vehicle_sim_data.py     # Feature engineering for neural networks
│   ├── sim_prepare_data.py             # Data restructuring scripts
│   ├── vehicle_sim_loader.py           # Custom PyTorch Dataset/DataLoader wrapper
│   ├── convert_vehicle_sim_mat_to_csv.py
│   └── sim_convert_mat_to_csv.py       # Converters for incoming simulator data
│
├── datasetforvehiclesimulation_csv/    # Pre-generated simulation test runs
│   ├── 1_RunTimeDataset_DLC_u85_v60.csv / .json  # Double Lane Change (High Friction)
│   ├── 5_RunTimeDataset_DLC_highSpeed.csv        # Double Lane Change (High Speed Edge Case)
│   ├── 6_RunTimeDataset_splitmu.csv              # Split-μ Braking Dynamic Transient Maneuver
│   └── 10_RunTimeDataset_Sine_accel.csv          # Sine-wave acceleration matrix
│
├── trainingdata/                       # 14 Real-sensor CSVs from TUM dataset (Train)
├── testingdata/                        # Real-sensor CSVs from TUM dataset (Test)
└── params/                             # Physical vehicle metrics & sensor noise profiles
    └── parameters.toml
```

## 🔄 Two-Stage Hybrid Execution Flow

```
The system processes raw, high-frequency physical data streams through a decoupled, two-stage framework to calculate unmeasurable vehicle states:
                   +---------------------------------------+
                   | Raw Vehicle Sensor Telemetry (.csv)   |
                   |  (IMU Accelerations, Wheel Speeds)    |
                   +-------------------+-------------------+
                                       |
                                       v
                   +---------------------------------------+
                   | STAGE 1: Signal Conditioning & EKF    |
                   |      (vehicle_force_ekf.py)           |
                   +-------------------+-------------------+
                                       |
                                       v  [Fused Kinematic States]
                   +---------------------------------------+
                   | STAGE 2: Deep Force Transformer       |
                   |     (force_transformer_model.pt)      |
                   +-------------------+-------------------+
                                       |
                                       v  [Virtual Sensor Inference]
                   +---------------------------------------+
                   | OUTPUT: Highly Nonlinear Forces &     |
                   | Dynamic Tire-Road Friction Coeff (μ)  |
                   +---------------------------------------+
```

### 1. Stage 1: Kinematic Extended Kalman Filtering (EKF)
* **The Script:** `vehicle_force_ekf.py` & `sim_force_ekf.py`
* **Mechanics:** Processes noisy raw physical data (longitudinal/lateral acceleration, yaw rate) to reliably track primary structural dynamics. It acts as a deterministic layer to stabilize base state estimates before feeding them into downstream neural models.

### 2. Stage 2: Deep Transformer Virtual Sensors 
* **The Script:** `force_transformer_common.py`
* **Mechanics:** Traditional linear tires fail to capture extreme transient slips. This stage utilizes a custom **PyTorch Transformer** to process temporal windows of state data. It accurately infers highly non-linear, physically unmeasurable metrics like individual tire-road force matrices and instantaneous friction coefficients ($\mu$).

---

## Dataset Profiles & Maneuvers
The system is built and cross-validated against diverse vehicle simulation test profiles located in `datasetforvehiclesimulation_csv/`:
* **Double Lane Change (DLC):** Evaluates high-speed lateral transient stability and weight transfer dynamics under different velocities (`v60`) and surface coefficients (`u85`).
* **Split-μ Braking:** Simulates severe braking conditions where left and right tires experience completely different grip levels, testing the fusion pipeline's robustness.
* **Sine Wave Acceleration:** Evaluates long-duration oscillatory inputs to identify phase lags and check for sensor drift accumulation within the filtering matrices.

---

## Step-by-Step Run Guide

###  Environment Setup
Ensure you have Python 3.10+ installed along with the required scientific computing and deep learning packages:

```bash
    pip install torch numpy pandas scikit-learn
```
# Vehicle Dynamics Workflows

This repo currently has two datasets and two estimator styles:

1. `trainingdata/` + `testingdata/` for the TUM real-sensor branch
2. `datasetforvehiclesimulation_csv/` for the converted simulation branch

The scripts in `codes_allFilters_architectures/` are the ones to use.

## What to run

### 1) TUM dataset with the new architecture

This is the real-sensor mini-transformer + EKF pipeline.

Train on the 14 training CSVs:

```powershell
& 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\nclt_env\Scripts\python.exe' `
  'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\train_newarch.py' `
  --dataset real `
  --train-folder 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\trainingdata' `
  --params-file 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\params\parameters.toml' `
  --vehicle-config 'real_sensor_vehicle_config.toml' `
  --model-out 'real_sensor_force_transformer.pt'
```

Run the trained model on the TUM test file:

```powershell
& 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\nclt_env\Scripts\python.exe' `
  'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\run_newarch.py' `
  --dataset real `
  --csv 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\testingdata\data_to_run.csv' `
  --params-file 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\params\parameters.toml' `
  --vehicle-config 'real_sensor_vehicle_config.toml' `
  --model 'real_sensor_force_transformer.pt' `
  --plot 'real_sensor_force_ekf.png'
```

You can also train on TUM and evaluate the same architecture on the converted simulation cases, as long as the simulation CSVs are first converted into the same sensor layout:

```powershell
& 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\nclt_env\Scripts\python.exe' `
  'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\convert_sim_to_tum_csv.py' `
  --folder 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\datasetforvehiclesimulation_csv' `
  --output-folder 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\sim_tumlike'

& 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\nclt_env\Scripts\python.exe' `
  'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\run_newarch.py' `
  --dataset real `
  --csv 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\sim_tumlike\1_RunTimeDataset_DLC_u85_v60.csv' `
  --params-file 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\params\parameters.toml' `
  --vehicle-config 'real_sensor_vehicle_config.toml' `
  --model 'real_sensor_force_transformer.pt' `
  --plot 'real_sensor_force_on_sim_case1.png'
```

This is the cleanest way to show how a TUM-trained new architecture behaves on simulation data without changing the estimator code.

### 2) Simulation dataset with the new architecture

This is the converted simulation mini-transformer + EKF pipeline.

Train on the 10 converted simulation CSVs:

```powershell
& 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\nclt_env\Scripts\python.exe' `
  'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\train_newarch.py' `
  --dataset sim `
  --folder 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\datasetforvehiclesimulation_csv' `
  --model-out 'sim_newarch_force_transformer.pt'
```

Run the trained model on one case:

```powershell
& 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\nclt_env\Scripts\python.exe' `
  'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\run_newarch.py' `
  --dataset sim `
  --csv 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\datasetforvehiclesimulation_csv\1_RunTimeDataset_DLC_u85_v60.csv' `
  --model 'sim_newarch_force_transformer.pt' `
  --plot 'sim_newarch_ekf_case1.png'
```

Run the same model on all 10 cases:

```powershell
Get-ChildItem 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\datasetforvehiclesimulation_csv' -Filter '*.csv' |
  Where-Object { $_.Name -ne 'scenario_summary.csv' } |
  ForEach-Object {
    & 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\nclt_env\Scripts\python.exe' `
      'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\run_newarch.py' `
      --dataset sim `
      --csv $_.FullName `
      --model 'sim_newarch_force_transformer.pt' `
      --plot ("{0}_newarch.png" -f $_.BaseName)
  }
```

### 3) TUM dataset with UKF, no transformer tire forces

This is the classical UKF baseline.

Run it directly on the TUM test file:

```powershell
& 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\nclt_env\Scripts\python.exe' `
  'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\run_ukf.py' `
  --csv 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\testingdata\data_to_run.csv' `
  --plot 'tum_ukf_real.png'
```

### 4) Simulation dataset with UKF

First convert each simulation CSV into a TUM-like CSV:

```powershell
& 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\nclt_env\Scripts\python.exe' `
  'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\convert_sim_to_tum_csv.py' `
  --folder 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\datasetforvehiclesimulation_csv' `
  --output-folder 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\sim_tumlike'
```

Then run the same UKF on one converted case:

```powershell
& 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\nclt_env\Scripts\python.exe' `
  'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\run_ukf.py' `
  --csv 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\sim_tumlike\1_RunTimeDataset_DLC_u85_v60.csv' `
  --plot 'tum_ukf_sim_case1.png'
```

Run the same UKF on all 10 converted simulation files:

```powershell
Get-ChildItem 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\sim_tumlike' -Filter '*.csv' |
  ForEach-Object {
    & 'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\nclt_env\Scripts\python.exe' `
      'D:\Downloads_D\Books\Research Papers\Vehicle dynamics\codes_allFilters_architectures\run_ukf.py' `
      --csv $_.FullName `
      --plot ("{0}_ukf.png" -f $_.BaseName)
  }
```

## Notes

- The UKF does not need training. It only needs the right input columns and tuned noise matrices.
- The new-architecture training scripts do need training because they use a mini-transformer to learn tire-force or vehicle-state mappings.
- `train_newarch.py` and `run_newarch.py` are the unified entrypoints. The dataset-specific scripts still exist underneath them, but you do not need to call them directly.
- For the simulation branch, `beta_true_rad` is kept only as a label for evaluation.
- For the TUM branch, the scripts read the real sensor CSVs directly.

## Important scripts

- `codes_allFilters_architectures/train_newarch.py`
- `codes_allFilters_architectures/run_newarch.py`
- `codes_allFilters_architectures/run_ukf.py`
- `codes_allFilters_architectures/tum_ukf.py`
- `codes_allFilters_architectures/convert_sim_to_tum_csv.py`

