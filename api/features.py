"""Transformaciones del notebook, empaquetadas para usarlas en producción.

Recibe préstamos "crudos" (las mismas columnas que X.csv) y devuelve las
matrices exactas que esperan el modelo de PD y el de LGD.

Todo lo que el modelo aprendió del train (percentiles, medianas, categorías,
orden de columnas) viene de `meta`, que se exporta desde el notebook.
"""
import numpy as np
import pandas as pd

from employment_sector import build_employment_features

PURPOSE_MAP = {
    'debt_consolidation': 'Debt Consolidation',
    'credit_card': 'Credit Card',
    'home_improvement': 'Home & Property',
    'house': 'Home & Property',
    'renewable_energy': 'Home & Property',
    'other': 'Other',
    'wedding': 'Other',
    'major_purchase': 'Major Purchase',
    'educational': 'Major Purchase',
    'medical': 'Medical',
    'car': 'Vehicle',
    'small_business': 'Small Business',
    'vacation': 'Personal & Lifestyle',
    'moving': 'Personal & Lifestyle',
}

LENGTH_MAP = {
    '< 1 year': 'Entry-Level',
    '1 year': 'Entry-Level',
    '2 years': 'Early Career',
    '3 years': 'Early Career',
    '4 years': 'Mid-Career',
    '5 years': 'Mid-Career',
    '6 years': 'Mid-Career',
    '7 years': 'Experienced',
    '8 years': 'Experienced',
    '9 years': 'Experienced',
    '10+ years': 'Long Tenure',
    'unknown length profile': 'Unknown Length Profile',
    'possible pensioner profile': 'Possible Pensioner Profile',
}

STATE_REGIONS = {
    **dict.fromkeys(['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'], 'Northeast'),
    **dict.fromkeys(['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'], 'Midwest'),
    **dict.fromkeys(['DE', 'DC', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS',
                     'TN', 'AR', 'LA', 'OK', 'TX'], 'South'),
    **dict.fromkeys(['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA'], 'West'),
}


def _grade_group_final(grade):
    g = grade.astype('string').str.strip().str.upper()
    letter = g.str.extract(r'^([A-G])', expand=False)
    number = pd.to_numeric(g.str.extract(r'^([A-G])([1-5])$', expand=True)[1], errors='coerce')
    group = pd.Series(
        np.select([number.isin([1, 2, 3]), number.isin([4, 5])],
                  [letter + '+', letter + '-'], default='Unknown'),
        index=g.index)
    return group.where(~letter.isin(['E', 'F', 'G']), 'EFG')


def _cuota_mensual(monto, tasa, plazo):
    """Fórmula de amortización (coincide con monthly_payment en el 99.96% de las filas)."""
    r = tasa / 1200
    return monto * r / (1 - (1 + r) ** -plazo)


def construir_base(raw, meta):
    """Variables comunes a los dos modelos (equivale a la limpieza del notebook)."""
    df = raw.copy()

    if 'monthly_payment' not in df or df['monthly_payment'].isna().any():
        calc = _cuota_mensual(df['funded_amnt'], df['interest_rate'], df['loan_term_months'])
        df['monthly_payment'] = df.get('monthly_payment', calc).fillna(calc)

    df['Loss'] = 100 * df['max_bal_owed'] / df['funded_amnt']

    df = build_employment_features(df)
    df['loan_purpose_group'] = (df['loan_purpose'].astype('string').str.strip().str.lower()
                                .map(PURPOSE_MAP).fillna('Other'))
    df['emp_length_group'] = (df['emp_length'].astype('string').str.strip().str.lower()
                              .map(LENGTH_MAP).fillna('Unknown Length Profile'))
    df['state_region'] = (df['addr_state'].astype('string').str.strip().str.upper()
                          .map(STATE_REGIONS).fillna('Unknown'))
    df['grade_group_final'] = _grade_group_final(df['grade'])

    df['debt_capacity'] = df['annual_income'] / df['funded_amnt']
    df['credit_age'] = df['issue_date_year'] - df['earliest_cr_line_year']
    df['had_delinq'] = df['mths_since_last_delinq'].notna().astype(int)
    df['payment_to_income'] = df['monthly_payment'] / (df['annual_income'] / 12)
    df['revol_bal_to_loan'] = df['total_credit_revolving_bal'] / df['funded_amnt']
    df['revol_bal_to_income'] = df['total_credit_revolving_bal'] / df['annual_income']

    for col in ['mths_since_last_installment_acc_op', 'bal_to_cred_lim']:
        df[f'{col}_missing'] = df[col].isna().astype(int)
        df[col] = df[col].fillna(meta['medianas'][col])

    # En el notebook esta bandera se creó después de imputar, así que en el
    # entrenamiento siempre vale 0. Se replica igual para no cambiar el modelo.
    df['no_installment_acc'] = df['mths_since_last_installment_acc_op'].isna().astype(int)
    return df


def _ordenar(X, cols, cats):
    faltan = [c for c in cols if c not in X.columns]
    if faltan:
        raise ValueError(f'Faltan columnas que el modelo espera: {faltan}')
    X = X[cols].copy()
    for c, categorias in cats.items():
        X[c] = pd.Categorical(X[c].astype('string'), categories=categorias)
    return X


def preparar_pd(base, meta):
    """Transformaciones que se aplicaron a X_train / X_test (clasificación)."""
    X = base.copy()
    X['dept_paym_income_ratio'] = X['dept_paym_income_ratio'].where(X['dept_paym_income_ratio'] >= 0)
    X[meta['clip_cols']] = X[meta['clip_cols']].clip(lower=meta['p01'], upper=meta['p99'], axis=1)
    X[meta['log_cols']] = np.log1p(X[meta['log_cols']])
    for c in meta['bin_cols']:
        X[c] = (X[c] > 0).astype(int)
    return _ordenar(X, meta['cols_pd'], meta['cats_pd'])


def preparar_lgd(base, meta):
    """El modelo de LGD se entrenó con dfR, sin log ni recorte."""
    return _ordenar(base, meta['cols_lgd'], meta['cats_lgd'])
