"""
Clustering API for Patient Risk Prediction
Uses pre-trained models from artifact/ directory
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# =============================================================================
# Configuration
# =============================================================================

ARTIFACT_DIR = Path(__file__).parent.parent / 'artifact'
CLUSTER_DIR = Path(__file__).parent.parent / 'cluster'

RANDOM_STATE = 42
API_HOST = "0.0.0.0"
API_PORT = 8001  # Different from RAG API (8000)

# =============================================================================
# Pydantic Models
# =============================================================================

class PatientData(BaseModel):
    """Input model for patient data"""
    # Numeric
    imc: float = Field(default=0.0)
    valor_de_ca125: float = Field(default=0.0)
    tamano_tumoral: float = Field(default=0.0)
    recep_est_porcent: float = Field(default=0.0)
    rece_de_Ppor: float = Field(default=0.0)
    edad_en_cirugia: float = Field(default=0.0)
    
    # Categorical (as strings)
    asa: str = Field(default="Missing")
    histo_defin: str = Field(default="Missing")
    grado_histologi: str = Field(default="Missing")
    FIGO2023: str = Field(default="Missing")
    afectacion_linf: str = Field(default="Missing")
    metasta_distan: str = Field(default="Missing")
    AP_centinela_pelvico: str = Field(default="Missing")
    AP_ganPelv: str = Field(default="Missing")
    AP_glanPaor: str = Field(default="Missing")
    beta_cateninap: str = Field(default="Missing")
    
    # Flags
    histo_defin__from_pre: int = Field(default=0)
    grado_histologi__from_pre: int = Field(default=0)
    FIGO2023__from_pre: int = Field(default=0)
    imc_miss: int = Field(default=0)
    tamano_tumoral_miss: int = Field(default=0)
    valor_de_ca125_miss: int = Field(default=0)
    
    class Config:
        extra = "ignore"


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    success: bool
    prediction: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PatientListResponse(BaseModel):
    """Response model for patient list"""
    success: bool
    patients: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


# =============================================================================
# Load Models and Data
# =============================================================================

# Feature schema
FEATURE_SCHEMA = None
EXPECTED_COLUMNS = []
NUM_COLS = []
CAT_COLS = []
FLAG_COLS = []
HORIZONS_DAYS = [365, 1095, 1825]

# Models
risk_model_rsf = None
cluster_assigner = None
subcluster1_assigner = None
SUBCLUSTER1_LABEL_MAP = {}

# Reference data
df_reference = None
RISK_TERCILES = None

# Category defaults
CAT_DEFAULT_BY_COL = {}


def load_artifacts():
    """Load all pre-trained models and schemas"""
    global FEATURE_SCHEMA, EXPECTED_COLUMNS, NUM_COLS, CAT_COLS, FLAG_COLS, HORIZONS_DAYS
    global risk_model_rsf, cluster_assigner, subcluster1_assigner, SUBCLUSTER1_LABEL_MAP
    global df_reference, RISK_TERCILES, CAT_DEFAULT_BY_COL
    
    print("Loading artifacts...")
    
    # Load feature schema
    schema_path = ARTIFACT_DIR / 'feature_schema.json'
    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found: {schema_path}")
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        FEATURE_SCHEMA = json.load(f)
    
    EXPECTED_COLUMNS = FEATURE_SCHEMA['expected_columns']
    NUM_COLS = FEATURE_SCHEMA['num_cols']
    CAT_COLS = FEATURE_SCHEMA['cat_cols']
    FLAG_COLS = FEATURE_SCHEMA['flag_cols']
    HORIZONS_DAYS = FEATURE_SCHEMA.get('horizons_days', [365, 1095, 1825])
    
    print(f"  Schema loaded: {len(EXPECTED_COLUMNS)} columns")
    
    # Load models
    risk_model_rsf = joblib.load(ARTIFACT_DIR / 'risk_model_rsf.joblib')
    cluster_assigner = joblib.load(ARTIFACT_DIR / 'cluster_assigner.joblib')
    subcluster1_assigner = joblib.load(ARTIFACT_DIR / 'subcluster1_assigner.joblib')
    
    print("  Models loaded: RSF, Cluster, Subcluster")
    
    # Load subcluster label map
    with open(ARTIFACT_DIR / 'subcluster1_label_map.json', 'r', encoding='utf-8') as f:
        SUBCLUSTER1_LABEL_MAP = json.load(f)
    
    # Set up category defaults
    CAT_DEFAULT_BY_COL = {c: 'Missing' for c in CAT_COLS}
    
    # Load reference data for risk terciles
    ref_candidates = [
        CLUSTER_DIR / 'df_con_subclusters.csv',
        CLUSTER_DIR / 'df_con_clusters.csv',
    ]
    
    for ref_path in ref_candidates:
        if ref_path.exists():
            df_reference = pd.read_csv(ref_path)
            if 'Unnamed: 0' in df_reference.columns:
                df_reference = df_reference.rename(columns={'Unnamed: 0': 'index'})
            print(f"  Reference data loaded: {ref_path.name} ({len(df_reference)} patients)")
            break
    
    if df_reference is not None:
        # Compute risk terciles
        try:
            X_ref = align_dataframe(df_reference)
            pred_ref = rsf_predict_survival_at_horizons(X_ref)
            r = pred_ref['risk_3y'].to_numpy(dtype=float)
            r = r[np.isfinite(r)]
            if len(r) >= 10:
                RISK_TERCILES = (
                    float(np.nanquantile(r, 1.0 / 3.0)),
                    float(np.nanquantile(r, 2.0 / 3.0))
                )
                print(f"  Risk terciles computed: q33={RISK_TERCILES[0]:.3f}, q66={RISK_TERCILES[1]:.3f}")
        except Exception as e:
            print(f"  Warning: Could not compute risk terciles: {e}")
    
    print("Artifacts loaded successfully!")


def align_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    """Align a DataFrame to the expected schema"""
    df = df_in.copy()
    
    # Add missing columns
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            if col in NUM_COLS:
                df[col] = 0.0
            elif col in CAT_COLS:
                df[col] = CAT_DEFAULT_BY_COL.get(col, 'Missing')
            elif col in FLAG_COLS:
                df[col] = 0
            else:
                df[col] = 0
    
    # Keep only expected columns, in order
    df = df.loc[:, EXPECTED_COLUMNS]
    
    # Normalize dtypes
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
    
    for col in FLAG_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    for col in CAT_COLS:
        df[col] = df[col].astype(object)
        df[col] = df[col].where(df[col].notna(), CAT_DEFAULT_BY_COL.get(col, 'Missing'))
        df[col] = df[col].astype(str)
    
    return df


def _step_fn_value_at(step_fn, t_days: float) -> float:
    """Evaluate a step function at a given time"""
    t = float(t_days)
    
    if hasattr(step_fn, 'x'):
        try:
            x = np.asarray(step_fn.x, dtype=float)
            if x.size > 0:
                t = min(t, float(x.max()))
        except Exception:
            pass
    
    try:
        return float(step_fn(t))
    except Exception:
        if hasattr(step_fn, 'x') and hasattr(step_fn, 'y'):
            x = np.asarray(step_fn.x, dtype=float)
            y = np.asarray(step_fn.y, dtype=float)
            if x.size == 0 or y.size == 0:
                return float('nan')
            idx = int(np.searchsorted(x, t, side='right') - 1)
            idx = max(0, min(idx, y.size - 1))
            return float(y[idx])
        raise


def rsf_predict_survival_at_horizons(X_aligned: pd.DataFrame) -> pd.DataFrame:
    """Predict survival probabilities at specified horizons"""
    preprocess = risk_model_rsf.named_steps['preprocess']
    model = risk_model_rsf.named_steps['model']
    
    X_t = preprocess.transform(X_aligned)
    sf_list = model.predict_survival_function(X_t)
    
    out = {'S_1y': [], 'S_3y': [], 'S_5y': [], 'risk_3y': []}
    t1, t3, t5 = HORIZONS_DAYS
    
    for sf in sf_list:
        s1 = _step_fn_value_at(sf, t1)
        s3 = _step_fn_value_at(sf, t3)
        s5 = _step_fn_value_at(sf, t5)
        out['S_1y'].append(s1)
        out['S_3y'].append(s3)
        out['S_5y'].append(s5)
        out['risk_3y'].append(float(1.0 - s3))
    
    return pd.DataFrame(out, index=X_aligned.index)


def risk_group_from_risk_3y(risk_3y: float) -> Optional[str]:
    """Determine risk group from 3-year risk"""
    if RISK_TERCILES is None or not np.isfinite(risk_3y):
        return None
    q33, q66 = RISK_TERCILES
    if risk_3y <= q33:
        return 'bajo'
    if risk_3y <= q66:
        return 'medio'
    return 'alto'


def _get_final_estimator(pipeline):
    """Get the final estimator from a pipeline"""
    if hasattr(pipeline, 'named_steps'):
        for key in ['clf', 'model', 'classifier', 'estimator']:
            if key in pipeline.named_steps:
                return pipeline.named_steps[key]
        return pipeline.steps[-1][1]
    return pipeline


def assign_cluster_and_subcluster(X_aligned: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Assign cluster and subcluster to patients"""
    out = pd.DataFrame(index=X_aligned.index)
    
    # Cluster prediction
    out['cluster_pred'] = cluster_assigner.predict(X_aligned).astype(int)
    
    # Cluster confidence
    out['p_cluster1'] = np.nan
    out['low_confidence_cluster'] = False
    
    if hasattr(cluster_assigner, 'predict_proba'):
        try:
            cluster_clf = _get_final_estimator(cluster_assigner)
            if hasattr(cluster_clf, 'classes_'):
                classes = np.asarray(cluster_clf.classes_)
                idx_1 = np.where(classes == 1)[0]
                if idx_1.size > 0:
                    proba = cluster_assigner.predict_proba(X_aligned)
                    out['p_cluster1'] = proba[:, int(idx_1[0])].astype(float)
                    out['low_confidence_cluster'] = out['p_cluster1'].notna() & (np.abs(out['p_cluster1'] - 0.5) < 0.10)
        except Exception as e:
            print(f"Warning: Could not compute cluster probability: {e}")
    
    # Subcluster defaults
    out['subcluster_pred'] = None
    out['p_subcluster11'] = np.nan
    out['threshold_used'] = np.nan
    out['low_confidence_flag'] = False
    
    # Assign subcluster for cluster 1 patients
    idx_cluster1 = out['cluster_pred'] == 1
    if idx_cluster1.any():
        o2m = SUBCLUSTER1_LABEL_MAP.get('original_to_model', {})
        
        if '11' in o2m:
            model_label_11 = int(o2m['11'])
            sub_clf = _get_final_estimator(subcluster1_assigner)
            
            if hasattr(sub_clf, 'classes_'):
                sub_classes = np.asarray(sub_clf.classes_)
                col_idx = int(np.where(sub_classes == model_label_11)[0][0])
                
                sub_proba = subcluster1_assigner.predict_proba(X_aligned.loc[idx_cluster1])
                p11 = sub_proba[:, col_idx].astype(float)
                
                out.loc[idx_cluster1, 'p_subcluster11'] = p11
                out.loc[idx_cluster1, 'threshold_used'] = float(threshold)
                
                orig_labels = SUBCLUSTER1_LABEL_MAP.get('original_labels_sorted', [11, 12])
                other_label = next((lbl for lbl in orig_labels if int(lbl) != 11), 12)
                out.loc[idx_cluster1, 'subcluster_pred'] = np.where(p11 >= threshold, 11, other_label)
                out.loc[idx_cluster1, 'low_confidence_flag'] = np.abs(p11 - 0.5) < 0.10
    
    return out


def _to_python_bool(val) -> bool:
    """Convert numpy.bool or any value to Python native bool"""
    if val is None:
        return False
    if isinstance(val, (bool, int)):
        return bool(val)
    # Handle numpy bool types
    try:
        return bool(val)
    except (TypeError, ValueError):
        return False


def predict_one(patient_dict: dict) -> dict:
    """Predict for a single patient"""
    df_one = pd.DataFrame([patient_dict])
    X_one = align_dataframe(df_one)
    
    # Compute warnings
    miss_cols = [c for c in EXPECTED_COLUMNS if c.endswith('_miss')]
    miss_sum = X_one[miss_cols].sum(axis=1).iloc[0] if miss_cols else 0
    many_missing_flag = _to_python_bool(miss_sum >= 3)
    
    # Get predictions
    profile = assign_cluster_and_subcluster(X_one).iloc[0].to_dict()
    surv = rsf_predict_survival_at_horizons(X_one).iloc[0].to_dict()
    risk_group = risk_group_from_risk_3y(float(surv['risk_3y']))
    
    return {
        'cluster_pred': int(profile['cluster_pred']),
        'p_cluster1': None if pd.isna(profile.get('p_cluster1')) else float(profile['p_cluster1']),
        'low_confidence_cluster': _to_python_bool(profile.get('low_confidence_cluster', False)),
        'subcluster_pred': int(profile['subcluster_pred']) if profile['subcluster_pred'] is not None else None,
        'p_subcluster11': None if pd.isna(profile['p_subcluster11']) else float(profile['p_subcluster11']),
        'threshold_used': None if pd.isna(profile['threshold_used']) else float(profile['threshold_used']),
        'S_1y': float(surv['S_1y']),
        'S_3y': float(surv['S_3y']),
        'S_5y': float(surv['S_5y']),
        'risk_3y': float(surv['risk_3y']),
        'risk_group': risk_group,
        'warnings': {
            'many_missing_flag': many_missing_flag,
            'low_confidence_cluster': _to_python_bool(profile.get('low_confidence_cluster', False)),
            'low_confidence_flag': _to_python_bool(profile.get('low_confidence_flag', False)),
            'unknown_category_flag': False,
        },
    }


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="NSMP Clustering API",
    description="API for patient clustering and survival prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Load models on startup"""
    try:
        load_artifacts()
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        raise


@app.get("/")
async def root():
    """Health check"""
    return {"status": "online", "service": "NSMP Clustering API"}


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": risk_model_rsf is not None,
        "reference_data_loaded": df_reference is not None,
        "patient_count": len(df_reference) if df_reference is not None else 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    """Predict cluster and survival for a patient"""
    try:
        patient_dict = patient.model_dump()
        prediction = predict_one(patient_dict)
        return PredictionResponse(success=True, prediction=prediction)
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return PredictionResponse(success=False, error=str(e))


@app.get("/patients", response_model=PatientListResponse)
async def get_patients():
    """Get list of all patients from reference data"""
    if df_reference is None:
        return PatientListResponse(success=False, error="No reference data loaded")
    
    try:
        # Compute predictions for all patients
        X_all = align_dataframe(df_reference)
        profile_df = assign_cluster_and_subcluster(X_all)
        surv_df = rsf_predict_survival_at_horizons(X_all)
        
        # Build patient list
        patients = []
        for i, row in df_reference.iterrows():
            idx = row.get('index', i)
            patient = {
                'index': int(idx) if not pd.isna(idx) else i,
                'edad_en_cirugia': row.get('edad_en_cirugia'),
                'FIGO2023': row.get('FIGO2023'),
                'cluster_pred': int(profile_df.loc[i, 'cluster_pred']),
                'risk_group': risk_group_from_risk_3y(surv_df.loc[i, 'risk_3y']),
            }
            patients.append(patient)
        
        return PatientListResponse(success=True, patients=patients)
    except Exception as e:
        print(f"Error getting patients: {e}")
        return PatientListResponse(success=False, error=str(e))


@app.get("/patients/{patient_index}")
async def get_patient(patient_index: int):
    """Get a specific patient by index"""
    if df_reference is None:
        raise HTTPException(status_code=404, detail="No reference data loaded")
    
    try:
        # Find patient
        if 'index' in df_reference.columns:
            patient_row = df_reference[df_reference['index'] == patient_index]
        else:
            patient_row = df_reference.iloc[[patient_index]] if patient_index < len(df_reference) else pd.DataFrame()
        
        if patient_row.empty:
            raise HTTPException(status_code=404, detail=f"Patient {patient_index} not found")
        
        patient_data = patient_row.iloc[0].to_dict()
        patient_data['index'] = patient_index
        
        # Get prediction
        prediction = predict_one(patient_data)
        
        return {
            "success": True,
            "patient": patient_data,
            "prediction": prediction
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting Clustering API server...")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
