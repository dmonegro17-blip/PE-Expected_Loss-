"""API de pérdida esperada (EL = PD × LGD × EAD).

Ejecutar:   uvicorn main:app --reload
Documentos: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from features import construir_base, preparar_lgd, preparar_pd

ARTEFACTOS = Path(__file__).parent / 'artefactos'
estado = {}


@asynccontextmanager
async def lifespan(app):
    # Los modelos se cargan una sola vez al arrancar, no en cada petición
    estado['meta'] = joblib.load(ARTEFACTOS / 'meta.joblib')
    estado['pd'] = xgb.XGBClassifier()
    estado['pd'].load_model(ARTEFACTOS / 'modelo_pd.json')
    estado['lgd'] = xgb.XGBRegressor()
    estado['lgd'].load_model(ARTEFACTOS / 'modelo_lgd.json')
    yield
    estado.clear()


app = FastAPI(title='API de Pérdida Esperada', version='1.0', lifespan=lifespan)


class Prestamo(BaseModel):
    """Datos de la solicitud. Son las columnas de X.csv que existen al aprobar el préstamo."""
    model_config = ConfigDict(populate_by_name=True)

    funded_amnt: float = Field(gt=0, description='Monto del préstamo (EAD)')
    interest_rate: float = Field(gt=0, lt=100)
    loan_term_months: Literal[36, 60]
    monthly_payment: float | None = Field(None, description='Si no se envía, se calcula con la fórmula de amortización')
    grade: str = Field(pattern=r'^[A-G][1-5]$')
    emp_title: str | None = None
    emp_length: str | None = None
    home_ownership_status: str
    annual_income: float = Field(gt=0)
    verification_status: str
    loan_purpose: str
    addr_state: str = Field(min_length=2, max_length=2)
    dept_paym_income_ratio: float
    num_30_delinq_in_2yrs: int = Field(ge=0, alias='num_30+_delinq_in_2yrs')
    num_inq_in_6mths: int = Field(ge=0)
    mths_since_last_delinq: float | None = None
    num_open_credit_lines: int = Field(ge=0)
    num_derogatory_pub_rec: int = Field(ge=0)
    total_credit_revolving_bal: float = Field(ge=0)
    used_credit_share: float = Field(ge=0)
    tot_num_credit_lines: int = Field(ge=0)
    initial_list_status: str
    num_open_trades_in_6mths: float = Field(ge=0)
    num_installment_acc_op_in_12mths: float = Field(ge=0)
    num_installment_acc_op_in_24mths: float = Field(ge=0)
    mths_since_last_installment_acc_op: float | None = None
    num_rev_trades_op_in_12mths: float = Field(ge=0)
    num_rev_trades_op_in_24mths: float = Field(ge=0)
    max_bal_owed: float = Field(ge=0)
    bal_to_cred_lim: float | None = None
    num_inq: float = Field(ge=0)
    num_inq_in_12mths: int = Field(ge=0)
    disbursement_method: str
    earliest_cr_line_year: int = Field(ge=1900)
    issue_date_year: int = Field(default_factory=lambda: date.today().year)


class Resultado(BaseModel):
    pd: float = Field(description='Probabilidad de impago (calibrada)')
    lgd: float = Field(description='Fracción de la deuda que se pierde si hay impago')
    ead: float = Field(description='Exposición en dólares')
    perdida_esperada: float = Field(description='EL = PD × LGD × EAD, en dólares')
    tasa_perdida: float = Field(description='EL / EAD')


def _predecir(prestamos: list[Prestamo]) -> list[Resultado]:
    meta = estado['meta']
    raw = pd.DataFrame([p.model_dump(by_alias=True) for p in prestamos])
    base = construir_base(raw, meta)

    p = estado['pd'].predict_proba(preparar_pd(base, meta))[:, 1]
    w = meta['w']
    pd_cal = p / (p + w * (1 - p))                       # deshacer scale_pos_weight

    lgd = np.clip(estado['lgd'].predict(preparar_lgd(base, meta)) / 100, 0, 1)
    ead = base['funded_amnt'].to_numpy(dtype=float)
    el = pd_cal * lgd * ead

    return [Resultado(pd=round(float(a), 4), lgd=round(float(b), 4), ead=float(c),
                      perdida_esperada=round(float(d), 2), tasa_perdida=round(float(d / c), 4))
            for a, b, c, d in zip(pd_cal, lgd, ead, el)]


@app.get('/salud')
def salud():
    return {'estado': 'ok', 'modelos_cargados': {'pd', 'lgd'} <= estado.keys()}


@app.post('/predecir', response_model=Resultado)
def predecir(prestamo: Prestamo):
    try:
        return _predecir([prestamo])[0]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post('/predecir_lote', response_model=list[Resultado])
def predecir_lote(prestamos: list[Prestamo]):
    if not 1 <= len(prestamos) <= 1000:
        raise HTTPException(status_code=422, detail='Envía entre 1 y 1000 préstamos')
    try:
        return _predecir(prestamos)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
