# Motor de Pérdida Esperada: Cartera Lending Club

> **Señores del Comité:** de una cartera de Test valorada en **3.767,7 millones de dólares**, nuestro motor estima que tendremos unas pérdidas por impago de **117,3 millones de dólares** (**3,11 %** de la exposición).

Proyecto final del curso de Data Science de Jyts. Simula el encargo del Chief Risk Officer de un banco que evalúa **comprar una cartera de préstamos al consumo de Lending Club**. El objetivo no es un buen *accuracy*, sino **una cifra en dólares**: cuánto dinero se espera perder.

$$EL = PD \times LGD \times EAD$$

| Componente | Qué es | Cómo se obtiene |
|---|---|---|
| **PD** (*Probability of Default*) | Probabilidad de impago de cada préstamo | Clasificador XGBoost, **calibrado** |
| **LGD** (*Loss Given Default*) | % de la deuda que se pierde si hay impago | Regresor XGBoost entrenado **solo con morosos**, sobre `Loss = 100 · max_bal_owed / funded_amnt` |
| **EAD** (*Exposure at Default*) | Dinero expuesto | `funded_amnt` (monto financiado) |

Además, el motor está desplegado como **API REST con FastAPI**, que devuelve PD, LGD y pérdida esperada en tiempo real para una solicitud nueva.

---

## Contenido

1. [Datos](#datos)
2. [Fase 1: EDA, limpieza y feature engineering](#fase-1-eda-limpieza-y-feature-engineering)
3. [Fase 2: Modelo de clasificación (PD)](#fase-2-modelo-de-clasificación-pd)
4. [Fase 3: Modelo de regresión (LGD)](#fase-3-modelo-de-regresión-lgd)
5. [Fase 4: Pérdida esperada e insights](#fase-4-pérdida-esperada-e-insights)
6. [¿Es explicable ante el regulador?](#es-explicable-ante-el-regulador)
7. [API en producción (FastAPI)](#api-en-producción-fastapi)
8. [Limitaciones y próximos pasos](#limitaciones-y-próximos-pasos)
9. [Cómo reproducirlo](#cómo-reproducirlo)

---

## Datos

- **Fuente:** [Lending Club Loan Data (Kaggle)](https://www.kaggle.com/datasets/db0boy/lending-club-loan-data-cleared/data)
- `X.csv`: **2.139.643 préstamos × 45 variables**. `target.csv`: impago (`y = 1`) o pagado (`y = 0`).
- Los dos archivos tienen el mismo número de filas y **se unen por posición** (no hay columna ID).
- Siguiendo el enunciado, se conservan solo los préstamos con `Loss < 300 %`. Este filtro también descarta las filas sin `max_bal_owed`. **Quedan 1.257.995 préstamos, con una tasa de impago del 10,17 %**, frente al 13,07 % del dataset completo.

> Los datos no están en el repositorio por su tamaño (450 MB). Descárgalos de Kaggle y colócalos en `data/`.

---

## Fase 1: EDA, limpieza y feature engineering

### Desbalanceo
Solo **1 de cada 10 préstamos** cae en impago. Un modelo que prediga siempre "paga" tendría un **89,8 % de accuracy** sin aprender nada. Por eso el accuracy se descarta y se usan **ROC-AUC, PR-AUC y Recall**.

### Variables eliminadas, y por qué

| Motivo | Variables | Justificación de negocio |
|---|---|---|
| **Fuga de información (*leakage*)** | `remaining_princ_for_tot_amnt_fund`, `princ_rec`, `paym_rec_for_tot_amnt_fund`, `interest_rec`, `late_fees_rec` | Son pagos **posteriores** a la concesión. El día que se aprueba el préstamo no existen. `remaining_princ` tenía por sí sola un AUC de 0,753, más que el tipo de interés: una señal de alarma. |
| **Sesgo temporal** | `issue_date_year`, `earliest_cr_line_year` | La tasa de impago cae de 18,6 % (2015) a 2,9 % (2018) porque los préstamos recientes **aún no han tenido tiempo de caer en impago**, no porque sean mejores. El modelo aprendería "reciente = seguro". Se sustituyen por `credit_age`. |
| **Redundantes** | `monthly_payment` | Se obtiene de monto, tasa y plazo con la fórmula de amortización (coincide en el 99,96 % de las filas). Se sustituye por `payment_to_income`. |
| | `mths_since_recent_bankcard_delinq`, `mths_since_recent_revol_delinq` | Correlación de 0,86–0,89 con `mths_since_last_delinq`. |
| **Sin señal** | `region_code` | AUC de 0,501; cada código mezcla 36–47 estados distintos. |
| **Alta cardinalidad** | `emp_title` | 263.133 valores distintos. Se reemplaza por `employment_sector`. |
| **Poco valor** | `issue_date_month`, `earliest_cr_line_month` | El mes no aporta información de riesgo. |

### Nulos
- `mths_since_last_delinq` (50 % de nulos): el nulo **significa "nunca ha estado en mora"**, así que no se borra. Se convierte en la bandera `had_delinq`.
- `mths_since_last_installment_acc_op` (3 %) y `bal_to_cred_lim` (119 filas): se imputan con la mediana y se añade una bandera `_missing`.
- `dept_paym_income_ratio = -1`: un ratio negativo no tiene sentido, así que se trata como nulo.

### Profesiones raras: `employment_sector`
`emp_title` es texto libre (por ejemplo "Manager", "manager", "mngr", "SHIPPING MANGER"). El módulo [`employment_sector.py`](notebooks/employment_sector.py):
1. **Imputa** los títulos vacíos con reglas de negocio: *Possible Pensioner* (sin empleo ni antigüedad, con ingresos e historial anterior a 1980), perfiles según la antigüedad laboral o *Unknown Employment Profile*.
2. **Normaliza** abreviaturas y errores de escritura (`mgr → manager`, `technition → technician`…).
3. **Clasifica** los títulos en **20 sectores**: *Management & Executive* (18,2 %), *Healthcare* (8,9 %), *Industrial & Skilled Trades* (8,4 %), etc.

### Agrupación de categorías

| Nueva variable | Origen | Grupos | Por qué |
|---|---|---|---|
| `loan_purpose_group` | `loan_purpose` (14) | 9 | Une propósitos pequeños con el mismo perfil de riesgo (por ejemplo *house*, *home_improvement* y *renewable_energy* en *Home & Property*). |
| `emp_length_group` | `emp_length` (11) | 7 | Etapas de carrera: *Entry-Level*, *Early Career*, *Mid-Career*, *Experienced*, *Long Tenure*. |
| `state_region` | `addr_state` (51) | 4 | Regiones del censo de EE. UU.; evita 51 categorías con pocos datos. |
| `grade_group_final` | `grade` (35) | 9 | `A1–A3 → A+`, `A4–A5 → A-`… Los grados E, F y G se unen en `EFG` porque juntos solo suman el 6,5 % de los préstamos. |

### Nuevas variables

| Variable | Fórmula | Qué mide |
|---|---|---|
| `payment_to_income` | cuota mensual / (ingreso anual / 12) | Qué parte del sueldo se va en la cuota |
| `debt_capacity` | ingreso anual / monto | Capacidad de pago frente al préstamo |
| `credit_age` | año de emisión − año de la primera línea de crédito | Antigüedad del historial crediticio |
| `had_delinq` | `mths_since_last_delinq` no nulo | Si alguna vez estuvo en mora |

### Valores extremos en ingresos
`annual_income` llega a **110 millones de dólares** (asimetría de 464). Para reducir su efecto sin borrar clientes:
- **Recorte** en los percentiles 1 y 99, **calculados solo con el train** para no filtrar información del test.
- **`log1p`** en las variables muy asimétricas: la asimetría de `annual_income` pasa de 464 a 0,24.
- **Binarización** de `num_derogatory_pub_rec` y `num_30+_delinq_in_2yrs` (más del 80 % son ceros).

---

## Fase 2: Modelo de clasificación (PD)

División **80/20 estratificada** (1.006.396 / 251.599 préstamos). Para el *early stopping* se separa además un 15 % del train como validación, de modo que **el test no se toca hasta la evaluación final**.

| Modelo | ROC-AUC | Gini | PR-AUC | Recall (clase 1) | Precisión (clase 1) |
|---|---|---|---|---|---|
| Regresión logística | 0,769 | 0,538 | 0,271 | 0,719 | 0,201 |
| HistGradientBoosting | 0,779 | 0,558 | 0,284 | 0,748 | 0,201 |
| XGBoost + RandomizedSearch, *con* `issue_date_year` | 0,783 | 0,566 | 0,288 | 0,734 | 0,209 |
| **XGBoost + RandomizedSearch, sin fechas** ✅ | **0,780** | **0,560** | **0,283** | **0,722** | **0,210** |

*PR-AUC al azar ≈ 0,10. Gini = 2 · AUC − 1, la convención bancaria.*

**Modelo elegido: XGBoost sin variables de fecha.** `issue_date_year` era la variable más importante del modelo anterior, pero su señal es el sesgo temporal descrito en la Fase 1. Quitarla **solo cuesta 0,003 de AUC**, y a cambio el modelo deja de depender de un efecto que no se repetirá con préstamos nuevos.

- **Desbalanceo:** `scale_pos_weight = 8,83` (ceros / unos).
- **Optimización:** `RandomizedSearchCV` con 25 combinaciones × 3 folds sobre una muestra de 200.000 préstamos, maximizando la PR-AUC. Mejores parámetros: `max_depth=7`, `learning_rate=0,0225`, `min_child_weight=129`, `subsample=0,66`, `colsample_bytree=0,80`, `gamma=2,51`, `reg_lambda=0,36`.
- **Variables más importantes (*gain*):** `grade_group_final`, `disbursement_method`, `interest_rate`, `num_rev_trades_op_in_24mths`, `verification_status`, `num_inq_in_6mths`, `num_30+_delinq_in_2yrs`.

### Recall frente a precisión: el umbral es una decisión de negocio
El CRO prefiere *"rechazar a un cliente regular que dejar pasar a un moroso"*. El umbral permite decidir ese equilibrio sin reentrenar:

| Umbral | Recall | Precisión | Lectura |
|---|---|---|---|
| 0,3 | **0,919** | 0,154 | Detecta 9 de cada 10 morosos, pero rechaza a muchos buenos clientes |
| 0,4 | 0,843 | 0,178 | |
| 0,5 | 0,722 | 0,210 | Equilibrio por defecto |
| 0,6 | 0,550 | 0,253 | |
| 0,7 | 0,341 | 0,315 | Rechaza menos, pero deja pasar a la mayoría de los morosos |

### Calibración: el paso clave
`scale_pos_weight` mejora la detección de morosos, pero **infla las probabilidades**: la PD media del modelo era **0,39**, cuando la tasa real es **0,10**. Usarla directamente habría multiplicado la pérdida esperada por casi 4. Se corrige deshaciendo el efecto del peso sobre los *odds*:

$$PD_{real} = \frac{p}{p + w\,(1-p)}, \qquad w = 8{,}83$$

---

## Fase 3: Modelo de regresión (LGD)

- **Target:** `Loss = 100 · max_bal_owed / funded_amnt`, la fórmula del enunciado.
- **Solo morosos:** se entrena con los **102.382 morosos del train de la PD** y se evalúa con los **25.595 morosos del test**. Así ningún préstamo del test de la Fase 4 ha sido visto por ninguno de los dos modelos.
- **Se excluye `max_bal_owed`** de las variables: junto con `funded_amnt` permitiría *calcular* `Loss` exactamente (R² ≈ 1 sin aprender nada).

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Línea base (predecir la media) | 26,18 | 39,21 | 0,000 |
| **XGBoost + RandomizedSearch** ✅ | **14,05** | **23,09** | **0,653** |

*MAE y RMSE en puntos porcentuales de `Loss`, medidos sobre los morosos del test.*

**Modelo elegido: XGBoost** (15 combinaciones × 3 folds, minimizando el RMSE). Se equivoca en promedio en **14 puntos**, frente a 26 de la línea base: reduce el error casi a la mitad. Mejores parámetros: `max_depth=9`, `learning_rate=0,026`, `min_child_weight=131`, `subsample=0,91`, `colsample_bytree=0,94`.

En una fase exploratoria se probó también una regresión Ridge con el target en logaritmo. Se descartó porque, al deshacer el logaritmo en los valores extremos, su RMSE se disparaba.

**Uso en la pérdida esperada:** la predicción se pasa a fracción (`/100`) y se **recorta en [0, 1]**, porque el banco no puede perder más del 100 % de lo que prestó.

---

## Fase 4: Pérdida esperada e insights

Se calcula sobre **todo el conjunto de Test de la PD**: 251.599 préstamos que ninguno de los dos modelos vio al entrenar.

### Cartera de Test

| Métrica | Valor |
|---|---|
| Préstamos | 251.599 |
| Exposición (EAD) | **$3.767.660.700** |
| Pérdida esperada (EL) | **$117.253.894** |
| Tasa de pérdida (EL / EAD) | **3,11 %** |
| PD media calibrada / tasa real | 9,34 % / 10,17 % |
| LGD media | 0,426 |

### Por grado de riesgo

| Grado | Préstamos | PD predicha | PD real | LGD | Tasa de pérdida | % de la EAD | % de la EL |
|---|---|---|---|---|---|---|---|
| A+ | 29.892 | 1,70 % | 1,83 % | 0,48 | **0,68 %** | 11,7 % | 2,6 % |
| A- | 22.613 | 3,03 % | 3,41 % | 0,46 | 1,17 % | 8,8 % | 3,3 % |
| B+ | 42.868 | 5,37 % | 6,00 % | 0,45 | 2,00 % | 16,3 % | 10,5 % |
| B- | 32.717 | 7,67 % | 8,62 % | 0,44 | 2,72 % | 12,2 % | 10,7 % |
| C+ | 45.888 | 10,03 % | 10,92 % | 0,42 | 3,37 % | 17,9 % | 19,4 % |
| C- | 27.863 | 12,55 % | 13,46 % | 0,39 | 4,01 % | 11,6 % | 14,9 % |
| D+ | 22.894 | 14,84 % | 16,29 % | 0,38 | 4,59 % | 9,5 % | 14,0 % |
| D- | 10.369 | 16,54 % | 17,52 % | 0,37 | 4,94 % | 4,4 % | 7,0 % |
| EFG | 16.495 | 25,90 % | 27,75 % | 0,34 | **7,21 %** | 7,6 % | **17,7 %** |

### Insights para el Comité

1. **La pérdida esperada es del 3,11 %.** De cada 100 dólares comprados se esperan perder 3,11, dentro del rango habitual del crédito al consumo (2–8 %).
2. **El riesgo crece de forma monótona con el grado.** La tasa de pérdida pasa de 0,68 % (A+) a 7,21 % (EFG), más de 10 veces. El grado asignado por Lending Club ordena bien el riesgo.
3. **La pérdida se concentra en los grados bajos.** Los grados E, F y G son solo el **7,6 % de la exposición**, pero generan el **17,7 % de la pérdida esperada**. Excluirlos o exigirles más rentabilidad reduce el riesgo de la compra de forma desproporcionada. C+ es el grado con más pérdida en dólares (22,8 millones), pero por volumen, no por riesgo.
4. **No calibrar cuesta caro.** Con la PD sin calibrar, la pérdida se habría sobreestimado unas 4 veces. Un error así puede llevar a rechazar una compra rentable.
5. **La PD subestima ligeramente, y de forma sistemática.** Predice 9,34 % frente al 10,17 % real, y queda por debajo en todos los grados. Con un ajuste prudente (× 1,089), la pérdida sería de unos **127,7 millones de dólares (3,39 %)**. **Recomendamos dotar la reserva con esta cifra.**

---

## ¿Es explicable ante el regulador?

XGBoost es más difícil de explicar que una regresión logística. La decisión se justifica así:

- **La ganancia es real pero moderada:** +1,1 puntos de AUC frente a la logística (0,780 frente a 0,769). La logística queda como **modelo de referencia (*challenger*)** explicable.
- **El modelo se puede auditar por sus resultados:** su PD sigue la tasa observada en cada grado y el riesgo crece de forma monótona, que es lo que un regulador revisa primero.
- **Sus variables principales tienen sentido de negocio:** el grado, el tipo de interés, la verificación de ingresos, las consultas de crédito recientes y las moras de los últimos 2 años.
- **Siguiente paso:** añadir **SHAP** para explicar cada decisión de forma individual ("¿por qué se rechazó a este cliente?"), un requisito habitual en el crédito al consumo.

---

## API en producción (FastAPI)

La carpeta [`api/`](api/) convierte el motor en un servicio: recibe los datos de una **solicitud nueva** y devuelve PD, LGD y pérdida esperada en milisegundos.

| Archivo | Función |
|---|---|
| [`main.py`](api/main.py) | API: valida la entrada (Pydantic), calibra la PD y calcula la EL |
| [`features.py`](api/features.py) | Repite **exactamente** las transformaciones del notebook |
| [`employment_sector.py`](api/employment_sector.py) | Clasificación de profesiones |
| [`ejemplo_prestamo.json`](api/ejemplo_prestamo.json) | Solicitud de ejemplo |

| Endpoint | Descripción |
|---|---|
| `GET /salud` | Comprueba que los modelos están cargados |
| `POST /predecir` | Un préstamo → `{pd, lgd, ead, perdida_esperada, tasa_perdida}` |
| `POST /predecir_lote` | Hasta 1.000 préstamos |

**Decisiones de diseño:**
- **La API solo pide datos que existen el día de la aprobación.** Las variables de *leakage* no forman parte de la entrada.
- Todo lo que el modelo aprendió del train (percentiles, medianas, categorías, orden de columnas, peso de calibración) se guarda en `meta.joblib` y no se escribe a mano.
- Las transformaciones de la API se **validan contra el notebook** con una prueba de paridad: se pasan filas crudas por `features.py` y se comprueba que la predicción coincide con la del notebook. En 2.000 préstamos de cada modelo, **la diferencia máxima fue 0,0** tanto en la PD como en la LGD.

```powershell
cd api
pip install -r requirements.txt
uvicorn main:app --reload
# Documentación interactiva: http://127.0.0.1:8000/docs
```

> Los modelos (`api/artefactos/`) no se suben por su tamaño. Se generan con la celda de exportación al final del notebook.

---

## Limitaciones y próximos pasos

- **La LGD es un *proxy*.** `max_bal_owed` es deuda en tarjetas de *antes* del préstamo, no recuperaciones reales tras el impago. Por eso la LGD **baja** en los grados de mayor riesgo (0,48 en A+, 0,34 en EFG), al contrario de lo que se esperaría de una LGD real. La pérdida de los grados bajos probablemente está **subestimada**.
- **`disbursement_method` también refleja la fecha.** El 91 % de los préstamos *DirectPay* son de 2018, así que parte de su señal (2.ª variable más importante) es el mismo sesgo temporal que el de `issue_date_year`. Dentro de 2018, *DirectPay* sigue teniendo menos impagos (1,0 % frente a 3,3 %), así que no es solo eso. Conviene evaluar el modelo sin ella.
- **Validación temporal.** El split es aleatorio. Una validación más exigente entrenaría con 2015–2017 y evaluaría con 2018, como ocurriría al comprar una cartera nueva.
- **La PD subestima ~8 %.** Una calibración isotónica o de Platt sobre el conjunto de validación podría corregirlo mejor que el ajuste proporcional.
- **Población:** el filtro `Loss < 300` deja fuera ~880.000 préstamos sin historial completo; el motor no aplica a ese perfil.
- **Producción real:** faltan autenticación, registro de predicciones, contenedor Docker y monitorización del *drift*.

---

## Cómo reproducirlo

```text
.
├── notebooks/
│   ├── Proyecto_Final_Expected_Loss.ipynb   # Fases 1–4 + correcciones finales + conclusiones
│   └── employment_sector.py                 # Clasificación de profesiones
├── api/                                     # FastAPI
├── data/                                    # X.csv y target.csv (no incluidos)
├── requirements.txt
└── README.md
```

1. Descarga los datos de Kaggle y colócalos en `data/`.
2. `pip install -r requirements.txt`
3. Ajusta las rutas de lectura en la primera celda del notebook y ejecútalo de principio a fin (**Restart & Run All**).

**Stack:** Python 3.13 · pandas 3.0 · scikit-learn · XGBoost 3.3 · FastAPI · Pydantic

---

*Diego Monegro · Proyecto final del curso de Data Science (Jyts)*
