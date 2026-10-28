"""
Clasificación de títulos laborales (emp_title) por sector.

Incluye todo el tratamiento de la información laboral:
  1. Imputación de emp_title y emp_length vacíos
     (pensionados, perfiles por antigüedad, desconocidos).
  2. Normalización de títulos (abreviaturas, errores de escritura,
     plurales, "senior/junior").
  3. Clasificación por reglas en cuatro niveles.
  4. Agrupación final de sectores.

Uso en un notebook (con este archivo en la misma carpeta):

    from employment_sector import build_employment_features
    df = build_employment_features(df)

Uso desde la terminal:

    python employment_sector.py entrada.csv salida.csv
    python employment_sector.py --check
"""

import argparse
import re

import pandas as pd


# ============================================================
# 1. IMPUTACIÓN DE TÍTULOS Y ANTIGÜEDAD VACÍOS
# ============================================================

EXPERIENCE_MAP = {
    "< 1 year": "Possible Entry-Level",
    "1 year": "Possible Entry-Level",
    "2 years": "Early-Career",
    "3 years": "Early-Career",
    "4 years": "Intermediate Experience",
    "5 years": "Intermediate Experience",
    "6 years": "Experienced Worker",
    "7 years": "Experienced Worker",
    "8 years": "Experienced Worker",
    "9 years": "Highly Experienced",
    "10+ years": "Long Tenure",
}


def impute_missing_titles(df):
    """Rellena emp_title y emp_length vacíos.

    Necesita las columnas emp_title, emp_length, annual_income y
    earliest_cr_line_year. Devuelve una copia del DataFrame.
    """
    df = df.copy()
    missing_title = df["emp_title"].isna()
    missing_length = df["emp_length"].isna()

    # 1. Posibles pensionados: sin título ni antigüedad, con ingresos
    #    y con historial crediticio anterior a 1980
    is_pensioner = (
        missing_title
        & missing_length
        & (df["annual_income"] > 1)
        & (df["earliest_cr_line_year"] < 1980)
    )
    df.loc[is_pensioner, "emp_title"] = "Possible Pensioner"
    df.loc[is_pensioner, "emp_length"] = "Possible Pensioner Profile"

    # 2. Resto de títulos vacíos según la antigüedad laboral
    still_missing = df["emp_title"].isna()
    df.loc[still_missing, "emp_title"] = (
        df.loc[still_missing, "emp_length"].map(EXPERIENCE_MAP)
    )

    # 3. Sin evidencia suficiente
    df["emp_title"] = df["emp_title"].fillna("Unknown Employment Profile")
    df["emp_length"] = df["emp_length"].fillna("Unknown Length Profile")
    return df


# ============================================================
# 2. NORMALIZAR LOS TÍTULOS LABORALES
# ============================================================

# Abreviaturas que se expanden antes de clasificar
ABBREVIATIONS = {
    "mgr": "manager", "mngr": "manager", "mgmt": "management",
    "pres": "president", "exec": "executive", "asst": "assistant",
    "supv": "supervisor", "supvr": "supervisor", "supt": "superintendent",
    "dir": "director", "engr": "engineer", "eng": "engineer",
    "coord": "coordinator", "spec": "specialist",
    "rep": "representative", "reps": "representative",
    "acct": "account", "accts": "accounts", "cust": "customer",
    "maint": "maintenance", "mech": "mechanic", "tec": "technician",
    "oper": "operator", "capt": "captain", "sgt": "sergeant",
    "lt": "lieutenant", "lt col": "lieutenant colonel",
}

# Errores de escritura y palabras mal unidas o separadas
TYPOS = {
    # manager / supervisor / superintendent / owner
    "manger": "manager", "maneger": "manager", "mananger": "manager",
    "mamager": "manager", "managet": "manager", "managee": "manager",
    "managment": "management",
    "superviser": "supervisor", "supervisior": "supervisor",
    "supevisor": "supervisor", "supervior": "supervisor",
    "suoervisor": "supervisor", "supervisory": "supervisor",
    "supervision": "supervisor",
    "superintendant": "superintendent", "super intendent": "superintendent",
    "super intendant": "superintendent",
    "own": "owner", "owne": "owner", "ownee": "owner", "ownet": "owner",
    "onwer": "owner", "ower": "owner", "owener": "owner", "0wner": "owner",
    "proprieter": "proprietor",
    # technician / maintenance
    "technition": "technician", "technitian": "technician",
    "technican": "technician", "techician": "technician",
    "technichian": "technician", "tecnician": "technician",
    "technicion": "technician", "techinician": "technician",
    "techican": "technician",
    "maintance": "maintenance", "maintainance": "maintenance",
    "maintence": "maintenance", "maintenence": "maintenance",
    "maintanance": "maintenance", "maintanence": "maintenance",
    "maintenace": "maintenance", "maintaince": "maintenance",
    "maintainence": "maintenance", "maintnance": "maintenance",
    "mantenance": "maintenance", "maitenance": "maintenance",
    "mainteance": "maintenance", "mantenimiento": "maintenance",
    # oficios
    "electrican": "electrician", "electrian": "electrician",
    "electrcian": "electrician", "electritian": "electrician",
    "electrition": "electrician", "electricain": "electrician",
    "machanic": "mechanic", "mecanic": "mechanic", "mecanico": "mechanic",
    "machinest": "machinist", "machinst": "machinist",
    "machnist": "machinist",
    "carpinter": "carpenter", "carpintero": "carpenter",
    "operater": "operator", "opperator": "operator",
    "operador": "operator", "opertor": "operator",
    "laboror": "laborer", "laberor": "laborer",
    "estimater": "estimator", "milwright": "millwright",
    "glazer": "glazier", "costodian": "custodian",
    "assambler": "assembler",
    # oficina, finanzas, ventas
    "adminstrator": "administrator", "administator": "administrator",
    "adminstrative": "administrative", "adminstration": "administration",
    "assitant": "assistant", "cordinator": "coordinator",
    "clerck": "clerk", "casher": "cashier",
    "bookeeper": "bookkeeper", "accoutant": "accountant",
    "acountant": "accountant", "anaylst": "analyst",
    "finacial": "financial", "controler": "controller",
    "contoller": "controller", "processer": "processor",
    "consulant": "consultant", "merchandizer": "merchandiser",
    "saleman": "salesman", "sale person": "salesperson",
    "costumer service": "customer service", "adviser": "advisor",
    "programer": "programmer", "enginner": "engineer",
    "enginer": "engineer",
    # otros
    "teachet": "teacher", "tracher": "teacher", "teacer": "teacher",
    "principle": "principal", "nirse": "nurse",
    "cosmotologist": "cosmetologist", "cheff": "chef", "sever": "server",
    "captian": "captain", "chauffer": "chauffeur",
    "dispacher": "dispatcher", "deiver": "driver", "drivet": "driver",
    "truck diver": "truck driver", "reciever": "receiver",
    "recieving": "receiving", "wharehouse": "warehouse",
    "warehose": "warehouse", "secuirty": "security",
    "sargent": "sergeant", "sargeant": "sergeant", "seargent": "sergeant",
    "attendent": "attendant",
    # palabras separadas que van juntas
    "house keeping": "housekeeping", "house keeper": "housekeeper",
    "book keeper": "bookkeeper", "book keeping": "bookkeeping",
    "handy man": "handyman", "bar tender": "bartender",
    "dish washer": "dishwasher", "journey man": "journeyman",
    "boiler maker": "boilermaker", "brick layer": "bricklayer",
    "grounds keeper": "groundskeeper", "care taker": "caretaker",
    "store keeper": "storekeeper", "iron worker": "ironworker",
    "door man": "doorman", "repair man": "repairman", "body man": "bodyman",
    "deck hand": "deckhand", "derrick hand": "derrickhand",
    "floor hand": "floorhand", "tool pusher": "toolpusher",
    "yard master": "yardmaster", "post master": "postmaster",
    "mill wright": "millwright", "black jack": "blackjack",
    "para professional": "paraprofessional",
    "para-professional": "paraprofessional",
    # palabras juntas que van separadas
    "teamleader": "team leader", "teamlead": "team lead",
    "groupleader": "group leader", "crewleader": "crew leader",
    "vicepresident": "vice president", "salesmanager": "sales manager",
    "asstmanager": "assistant manager", "lettercarrier": "mail carrier",
    "mailcarrier": "mail carrier", "busdriver": "bus driver",
    "casemanager": "case manager", "foodserver": "food server",
    "dockworker": "dock worker", "autoworker": "auto worker",
    "fulltime": "full time",
}

FIXES = {**ABBREVIATIONS, **TYPOS}
FIX_PATTERN = r"\b(?:" + "|".join(
    re.escape(key) for key in sorted(FIXES, key=len, reverse=True)
) + r")\b"

# Plurales en -men: foremen -> foreman, linemen -> lineman
# (sin tocar "specimen")
PLURAL_MEN_PATTERN = r"(?<!speci)\Bmen\b"

# Palabras de nivel que no indican sector:
# "senior advisor" -> "advisor", "coder ii" -> "coder"
SENIORITY_PATTERN = r"\b(?:senior|sr|junior|jr)\b|\s(?:i{1,3}|iv|[1-4])$"


def normalize_titles(titles):
    return (
        titles
        .astype(str)
        .str.lower()
        .str.replace(".", "", regex=False)   # r.n. -> rn, v.p. -> vp
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.replace(PLURAL_MEN_PATTERN, "man", regex=True)
        .str.replace(FIX_PATTERN, lambda m: FIXES[m.group(0)], regex=True)
        .str.replace(SENIORITY_PATTERN, "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ============================================================
# 3. REGLAS POR SECTOR
#
# Cada sector puede tener cuatro niveles. Se evalúan por NIVEL,
# no por sector (primero todos los "exact", luego todos los
# "specific", etc.):
#
#   exact    -> el título completo es exactamente ese texto
#   specific -> frases que deben ganarle a palabras clave de
#               OTRO sector (ej. "air traffic controller" frente
#               a "controller" de Finanzas)
#   keywords -> vocabulario propio del sector
#   generic  -> palabras de rol sin sector claro (manager,
#               specialist, clerk...). Solo se usan si nada más
#               coincidió.
#
# Dentro de un mismo nivel gana el sector que aparece antes.
# Los términos van en minúsculas, en singular (los plurales con
# -s/-es se reconocen solos), sin puntos, sin abreviaturas y sin
# "senior/junior" (eso ya se limpia al normalizar).
# ============================================================

SECTORS = {

    # --------------------------------------------------------
    # PERFILES DESCONOCIDOS E INFERIDOS
    # Incluye títulos que no dicen nada del trabajo.
    # --------------------------------------------------------
    "Unknown Employment Profile": {
        "exact": [
            "unknown employment profile", "unknown other", "full time",
            "full-time", "hourly", "temp", "overnight", "former", "general",
            "intern", "homemaker", "1",
        ],
    },
    "Inferred Profile": {
        "exact": [
            "possible pensioner", "possible entry-level", "early-career",
            "intermediate experience", "experienced worker",
            "highly experienced", "long tenure", "possible worker",
        ],
    },

    # --------------------------------------------------------
    # FINANZAS Y CONTABILIDAD
    # --------------------------------------------------------
    "Finance & Accounting": {
        "exact": ["processor", "advisor", "examiner", "closer", "account", "ap", "ar"],
        "keywords": [
            "cfo", "accountant", "accounting", "staff account", "controller",
            "comptroller", "auditor", "audit", "bookkeeper", "bookkeeping",
            "cpa", "actuary", "finance", "financial", "wealth", "banker",
            "banking", "bank", "teller", "underwriter", "underwriting",
            "loan", "lender", "lending", "mortgage", "mlo", "originator",
            "funder", "post closer", "credit", "tax", "revenue", "assessor",
            "payroll", "timekeeper", "billing", "biller", "payment poster",
            "cash poster", "treasurer", "treasury", "collector", "collection",
            "escrow", "investment", "investor", "stockbroker", "paraplanner",
            "trader", "trading", "appraiser", "appraisal", "title examiner",
            "title officer", "title searcher", "title processor",
            "abstractor", "trust officer", "client advisor",
            "registered principal", "voucher examiner", "loss mitigation",
            "risk", "bsa", "claim", "adjuster", "fiscal", "budget",
            "accounts payable", "accounts receivable", "account payable",
            "account receivable", "a/p", "a/r",
        ],
    },

    # --------------------------------------------------------
    # SALUD
    # "coder" / "coding" = codificación médica, no programación
    # --------------------------------------------------------
    "Healthcare": {
        "exact": [
            "ma", "pa", "cra", "mt", "rt", "ot", "pt", "don", "cpt", "lab",
            "resident", "nutrition",
        ],
        "keywords": [
            "rn", "nurse", "nursing", "np", "fnp", "aprn", "arnp", "crnp",
            "cnp", "rnfa", "lpn", "lvn", "cna", "stna", "lna", "gna", "rna",
            "cma", "ccma", "rma", "crna", "pca", "pct", "pta", "lpta", "rda",
            "rdh", "cda", "efda", "cota", "coa", "cst", "cpht", "psr",
            "rrt", "mlt", "cls", "chha", "ccht", "cht", "cmt", "rcis",
            "rpsgt", "cpc", "huc", "adon", "cno", "aemt", "ems", "bmet",
            "biomed", "medtech", "rvt", "cvt", "lvt", "pa-c", "physician",
            "doctor", "md", "hospitalist", "pediatrician", "pediatric",
            "cardiologist", "ophthalmologist", "podiatrist", "dds",
            "dentist", "dental", "hygienist", "pharmacist", "pharmacy",
            "therapist", "psychotherapist", "therapy", "medic", "paramedic",
            "emt", "phlebotomist", "phlebotomy", "veterinarian",
            "veterinary", "vet tech", "vet technician", "vet assistant",
            "psychologist", "neuropsychologist", "psychiatrist",
            "pathologist", "slp", "surgeon", "surgical", "surgery",
            "anesthesiologist", "anesthetist", "perfusionist", "radiologist",
            "radiology", "radiologic", "radiographer", "mammographer",
            "x-ray", "xray", "sonographer", "ultrasonographer",
            "echocardiographer", "ultrasound", "mri", "ekg",
            "nuclear medicine", "dosimetrist", "optometrist", "optician",
            "ophthalmic", "chiropractor", "acupuncturist", "orthotist",
            "dietitian", "dietician", "nutritionist", "epidemiologist",
            "epidemiology", "toxicologist", "physiologist",
            "neurophysiologist", "embryologist", "histotechnologist",
            "histotechnician", "histotech", "histologist",
            "cytotechnologist", "technologist", "dialysis", "respiratory",
            "audiologist", "midwife", "medical", "clinical", "clinician",
            "hospital", "healthcare", "health care", "public health",
            "patient", "admitting", "specimen", "infection preventionist",
            "home health", "hha", "health aid", "lab technician", "lab tech",
            "laboratory", "labor and delivery", "l&d", "athletic trainer",
            "mental health", "behavioral health", "behaviorist",
            "behavior interventionist", "behavioral interventionist", "bcba",
            "transcriptionist", "coder", "coding",
        ],
        "generic": ["aide", "aid"],
    },

    # --------------------------------------------------------
    # EDUCACIÓN Y FORMACIÓN
    # --------------------------------------------------------
    "Education & Training": {
        "exact": ["principal", "para", "admissions", "ta", "gra", "fellow", "teach"],
        "specific": ["school psychologist", "financial aid"],
        "keywords": [
            "teacher", "teaching", "professor", "educator", "instructor",
            "faculty", "adjunct", "lecturer", "dean", "tutor", "mentor",
            "substitute", "student", "graduate assistant", "librarian",
            "archivist", "curator", "historian", "registrar",
            "paraprofessional", "paraeducator", "preschool", "elementary",
            "education", "educational", "diagnostician", "trainer",
            "training", "facilitator", "coach", "athletics",
            "school counselor", "guidance counselor", "school principal",
            "assistant principal", "vice principal", "head of school",
            "school administrator", "school superintendent",
            "department chair", "program chair", "academic advisor",
            "college advisor", "career advisor", "enrollment advisor",
            "instructional aide", "admissions counselor",
            "admissions advisor", "admissions representative",
        ],
    },

    # --------------------------------------------------------
    # ANALÍTICA Y NEGOCIOS
    # --------------------------------------------------------
    "Business & Analytics": {
        "exact": ["ba", "analysis"],
        "keywords": [
            "business analyst", "business analysis", "data analyst",
            "data analysis", "research analyst", "operations analyst",
            "systems analyst", "market research", "statistical analyst",
            "business intelligence", "analytics", "statistician",
            "biostatistician", "economist", "project controls",
            "interviewer",
        ],
        "generic": ["analyst"],
    },

    # --------------------------------------------------------
    # TECNOLOGÍA, CIENCIA E INGENIERÍA
    # --------------------------------------------------------
    "Technology & Engineering": {
        "exact": ["qa", "technical", "mechanical", "research", "genius"],
        "keywords": [
            "it", "information technology", "information systems",
            "information assurance", "cto", "cio", "ciso", "isso",
            "software", "developer", "programmer", "programming",
            "webmaster", "scrum master", "engineer", "engineering",
            "technology", "network", "database", "dba", "devops",
            "sysadmin", "systems administrator", "system administrator",
            "systems admin", "system admin", "server administrator",
            "systems integrator", "computer", "telecom",
            "telecommunications", "help desk", "helpdesk", "service desk",
            "desktop support", "technical support", "tech support",
            "application support", "product support", "technical service",
            "technical staff", "technical expert", "technical advisor",
            "field service", "fsr", "fse", "information security",
            "cyber security", "cybersecurity", "security analyst",
            "quality assurance", "qa analyst", "qa tester", "tester",
            "product development", "r&d", "scientist", "researcher",
            "research associate", "research assistant", "research fellow",
            "postdoctoral", "postdoc", "chemist", "biologist",
            "microbiologist", "physicist", "geologist", "geophysicist",
            "hydrologist", "hydrogeologist", "meteorologist", "metrologist",
            "archaeologist", "cartographer", "architect",
            "architectural designer", "drafter", "draftsman", "cad",
            "steel detailer", "rebar detailer", "structural detailer",
            "surveyor", "surveying",
        ],
        "generic": ["technician", "tech"],
    },

    # --------------------------------------------------------
    # LEGAL
    # --------------------------------------------------------
    "Legal": {
        "specific": ["legal transcriptionist"],
        "keywords": [
            "attorney", "lawyer", "paralegal", "counsel", "legal", "law",
            "judge", "magistrate", "prosecutor", "public defender",
            "litigation", "compliance", "patent", "court clerk",
            "court reporter",
        ],
    },

    # --------------------------------------------------------
    # MILITAR
    # Va antes de Seguridad Pública para que "army sergeant"
    # o "navy officer" no caigan allí.
    # --------------------------------------------------------
    "Military": {
        "exact": ["major", "corporal", "commander", "marine", "mp"],
        "keywords": [
            "army", "navy", "naval", "air force", "usaf", "marines",
            "marine corps", "usmc", "usn", "coast guard", "national guard",
            "military", "soldier", "sailor", "airman", "infantry",
            "infantryman", "corpsman", "yeoman", "enlisted", "nco", "ncoic",
            "active duty", "platoon", "company commander",
            "commissioned officer", "staff officer", "colonel",
            "lance corporal", "staff sergeant", "master sergeant",
            "first sergeant", "sergeant major", "gunnery sergeant",
            "sergeant first class", "supply sergeant", "petty officer",
            "warrant officer",
            # Rangos abreviados
            "pfc", "spc", "ssg", "sfc", "sgm", "1sg", "ssgt", "tsgt",
            "msgt", "smsgt", "cmsgt", "lpo", "lcpo", "cpo",
            *[f"po{n}" for n in range(1, 4)],
            *[f"cw{n}" for n in range(2, 6)],
            # Grados de tropa: e1...e9 y e-1...e-9
            *[f"e{n}" for n in range(1, 10)],
            *[f"e-{n}" for n in range(1, 10)],
        ],
    },

    # --------------------------------------------------------
    # SEGURIDAD PÚBLICA Y SEGURIDAD LABORAL
    # --------------------------------------------------------
    "Public Safety": {
        "exact": ["officer", "co", "co1", "coii", "leo", "pso"],
        "specific": ["law enforcement", "fire engineer"],
        "keywords": [
            "police", "policeman", "sheriff", "deputy", "trooper",
            "detective", "investigator", "investigation", "criminalist",
            "csi", "special agent", "federal agent", "federal officer",
            "cbp", "cbpo", "customs officer", "custom protection",
            "border protection", "deportation", "immigration officer",
            "enforcement", "patrol", "patrolman", "ranger", "bailiff",
            "sergeant", "lieutenant", "corrections", "correctional",
            "correction officer", "detention", "jail", "probation",
            "parole", "community supervision officer", "warden", "jailer",
            "marshal", "firefighter", "fireman", "fire", "lifeguard",
            "security", "guard", "armed officer", "screener", "tsa", "tso",
            "loss prevention", "asset protection", "surveillance",
            "public safety", "safety", "animal control", "canine", "k9",
            "court officer", "peace officer", "code officer",
            "traffic officer", "community service officer",
            "school resource officer", "communications officer",
            "telecommunicator", "911 dispatcher", "911 dispatch",
            "emergency dispatcher", "emergency dispatch",
        ],
    },

    # --------------------------------------------------------
    # GOBIERNO Y ADMINISTRACIÓN PÚBLICA
    # --------------------------------------------------------
    "Government & Public Administration": {
        "keywords": [
            "government", "civil service", "civil servant", "public service",
            "public servant", "federal employee", "state employee", "mayor",
            "commissioner", "legislative", "legislator", "diplomat",
            "foreign service", "policy advisor", "policy analyst",
            "public affairs", "public information", "public administration",
            "city planner", "urban planner", "environmental planner",
            "program officer", "hearing officer", "decision review officer",
            "adjudicator", "ombudsman", "election", "asylum officer",
            "immigration service", "intelligence", "disability examiner",
            "benefit authorizer", "agpa", "ssa",
            # Grados federales: gs-1...gs-15 y gs1...gs15
            *[f"gs-{n}" for n in range(1, 16)],
            *[f"gs{n}" for n in range(1, 16)],
        ],
    },

    # --------------------------------------------------------
    # SERVICIOS SOCIALES Y COMUNITARIOS
    # --------------------------------------------------------
    "Social & Community Services": {
        "exact": ["provider"],
        "specific": ["salvation army"],
        "keywords": [
            "social worker", "social work", "social service", "lcsw",
            "case worker", "caseworker", "case manager", "case planner",
            "eligibility", "welfare", "texas works", "child support",
            "family service", "family support", "family worker",
            "protective service", "human service", "community outreach",
            "community liaison", "community health worker", "organizer",
            "youth worker", "foster", "direct care", "direct support",
            "direct service", "dsp", "qidp", "intake", "counselor",
            "advocate", "caregiver", "care giver", "caretaker",
            "care provider", "service provider", "companion", "home care",
            "homecare", "home aid", "home help", "home attendant",
            "home provider", "home visitor", "ihss", "child care",
            "childcare", "daycare", "pastor", "minister", "ministry",
            "clergy", "chaplain", "priest", "rabbi", "reverend", "rector",
            "missionary",
        ],
    },

    # --------------------------------------------------------
    # RECURSOS HUMANOS Y COORDINACIÓN
    # --------------------------------------------------------
    "Human Resources & Coordination": {
        "exact": ["generalist"],
        "keywords": [
            "hr", "hrbp", "chro", "human resource", "recruiter",
            "recruiting", "recruitment", "talent acquisition", "staffing",
            "personnel", "hris", "benefit", "labor relations",
        ],
        "generic": ["coordinator"],
    },

    # --------------------------------------------------------
    # TRANSPORTE Y CORREO
    # --------------------------------------------------------
    "Transportation": {
        "exact": ["drive", "fa", "navigator"],
        "specific": [
            "air traffic controller", "flight instructor",
            "locomotive engineer", "train engineer", "railroad engineer",
            "coach operator",
        ],
        "keywords": [
            "driver", "truckdriver", "truck drive", "driving", "trucker",
            "trucking", "teamster", "otr", "chauffeur", "courier",
            "messenger", "delivery", "transport", "transporter",
            "transportation", "freight", "hauler", "car hauler",
            "dispatcher", "dispatch", "pilot", "first officer", "aviation",
            "airline", "aircraft", "flight attendant", "flight crew",
            "inflight", "air traffic control", "atcs", "atc", "loadmaster",
            "ramp", "fleet service", "ground operations", "ground ops",
            "airport operations", "skycap", "baggage", "fueler",
            "gate agent", "ticket agent", "captain", "mate", "deckhand",
            "seaman", "mariner", "merchant marine", "tankerman",
            "conductor", "railroad", "carman", "trainman", "trainmaster",
            "yardmaster", "switchman", "signalman", "motorman",
            "track worker", "trackworker", "trackman", "hostler",
            "bus operator", "bus monitor", "bus attendant",
            "train operator", "owner operator", "parking", "lot attendant",
            "carrier", "rca", "pse", "ptf", "mail", "mailman", "mailhandler",
            "postal", "postmaster", "usps",
        ],
    },

    # --------------------------------------------------------
    # MANUFACTURA Y PRODUCCIÓN
    # --------------------------------------------------------
    "Manufacturing & Production": {
        "exact": ["machine", "cutter", "filler"],
        "specific": ["quality auditor"],
        "keywords": [
            "production", "manufacturing", "manufacturer", "assembly",
            "assembler", "fabricator", "fabrication", "factory",
            "plant worker", "line worker", "auto worker", "mill worker",
            "millworker", "machine operator", "machine op",
            "machine operation", "machine attendant", "machining", "cnc",
            "tender", "set up", "toolmaker", "tool and die", "tool & die",
            "die setter", "packaging", "pressman", "press operator",
            "printing", "printer", "prepress", "bindery", "laminator",
            "coater", "winder", "caster", "molder", "grinder", "polisher",
            "plater", "bonder", "engraver", "batcher", "blender", "mixer",
            "compounder", "feeder", "bottler", "brewer", "brewmaster",
            "winemaker", "miller", "sawyer", "saw filer", "finishing",
            "final finish", "presser", "seamstress", "tailor", "sewing",
            "sewer", "weaver", "upholsterer", "upholster", "upholstery",
            "quality", "qc",
        ],
        "generic": ["maker"],
    },

    # --------------------------------------------------------
    # ALMACÉN Y LOGÍSTICA
    # --------------------------------------------------------
    "Logistics & Warehousing": {
        "exact": ["stock", "supply", "material", "scanner", "processing"],
        "keywords": [
            "warehouse", "warehouseman", "warehousing", "logistics",
            "logistician", "supply chain", "supply officer", "shipping",
            "shipper", "receiving", "receiver", "inventory", "cycle counter",
            "stocker", "stocking", "stock clerk", "stock associate",
            "storekeeper", "cap team", "material handler",
            "materials handler", "material handling", "material control",
            "material processor", "merchandise processor",
            "merchandise handler", "order processor", "forklift",
            "distribution", "fulfillment", "package", "packing", "packer",
            "packager", "palletizer", "picker", "picking", "order puller",
            "puller", "loader", "unloader", "preloader", "stacker", "stower",
            "sorter", "selector", "order filler", "orderfiller", "mover",
            "dock", "dock worker", "expeditor", "expediter", "router",
            "weighmaster", "load planner", "longshoreman", "longshore",
            "stevedore", "demand planner", "supply planner",
            "material planner", "materials planner",
        ],
        "generic": ["handler"],
    },

    # --------------------------------------------------------
    # AGRICULTURA Y GANADERÍA
    # --------------------------------------------------------
    "Agriculture": {
        "keywords": [
            "farm", "farmer", "farming", "farmworker", "farmhand", "ranch",
            "rancher", "agriculture", "agricultural", "agronomist",
            "livestock", "grower", "forester", "forestry", "conservationist",
            "applicator",
        ],
    },

    # --------------------------------------------------------
    # OFICIOS, INSTALACIONES Y MANTENIMIENTO
    # --------------------------------------------------------
    "Facilities & Skilled Trades": {
        "exact": ["worker", "super", "environmental", "detail"],
        "specific": [
            "building engineer", "maintenance engineer",
            "stationary engineer", "garbage collector", "refuse collector",
            "fire alarm", "fire sprinkler",
        ],
        "keywords": [
            "foreman", "forman", "leadman", "lead man", "leadperson",
            "journeyman", "craftsman", "tradesman", "skilled trade",
            "construction", "builder", "contractor", "subcontractor",
            "laborer", "labor", "general worker", "maintenance",
            "maintainer", "facilities", "facility", "building service",
            "building official", "plans examiner", "public works",
            "handyman", "repairman", "repairer", "repair", "serviceman",
            "electrician", "electrical", "electric", "wireman", "plumber",
            "plumbing", "carpenter", "cabinet maker", "cabinetmaker",
            "woodworker", "framer", "drywall", "taper", "plasterer",
            "plaster", "bricklayer", "mason", "tile setter", "roofer",
            "glazier", "insulator", "insulation", "painter", "finisher",
            "installer", "installation", "install", "estimator",
            "estimating", "inspection", "hvac", "welder", "welding",
            "machinist", "millwright", "mechanic", "bodyman", "body shop",
            "fitter", "pipefitter", "steamfitter", "shipfitter",
            "pipe layer", "pipeliner", "boilermaker", "ironworker",
            "steel worker", "steelworker", "rigger", "rigging",
            "sheet metal", "sheetmetal", "elevator", "heavy equipment",
            "flagger", "highway", "locksmith", "lineman", "lineworker",
            "splicer", "locator", "meter reader", "utility", "utilities",
            "miner", "mining", "driller", "drilling", "roughneck",
            "roustabout", "derrickhand", "derrickman", "floorhand",
            "floorman", "toolpusher", "pusher", "pumper", "oiler", "blaster",
            "mwd", "diver", "auto technician", "auto tech",
            "automotive technician", "diesel technician", "detailer",
            "custodian", "custodial", "janitor", "janitorial", "cleaner",
            "cleaning", "porter", "doorman", "evs", "environmental service",
            "sanitation", "groundskeeper", "greenskeeper", "groundman",
            "groundsman", "grounds", "landscaper", "landscaping",
            "landscape", "gardener", "arborist", "tree", "exterminator",
            "pest control", "sprayer",
        ],
        "generic": ["operator", "inspector", "apprentice", "helper", "service worker"],
    },

    # --------------------------------------------------------
    # SERVICIO AL CLIENTE Y COMERCIO MINORISTA
    # --------------------------------------------------------
    "Customer Service & Retail": {
        "exact": ["service"],
        "keywords": [
            "customer", "csr", "csa", "csm", "css", "cssr", "csc", "ccr",
            "msr", "rsa", "call center", "client service", "client relations",
            "client support", "member service", "guest service",
            "front desk", "front end", "greeter", "cashier", "cash office",
            "checker", "bagger", "retail", "grocery", "produce",
            "bookseller", "personal shopper", "key holder", "keyholder",
            "store associate", "store worker", "sales associate",
            "sales clerk", "team member", "retention", "mobile expert",
            "at home advisor", "telephone operator", "switchboard operator",
        ],
        "generic": ["clerk"],
    },

    # --------------------------------------------------------
    # VENTAS, COMPRAS Y MARKETING
    # --------------------------------------------------------
    "Sales & Marketing": {
        "exact": ["sale", "ambassador"],
        "specific": ["medical sales", "software sales"],
        "keywords": [
            "sales", "salesman", "salesperson", "saleswoman", "telesales",
            "telemarketer", "seller", "vendor", "merchant", "marketer",
            "marketing", "advertising", "social media", "media planner",
            "content strategist", "digital strategist", "brand ambassador",
            "beauty advisor", "product advisor", "energy advisor",
            "account executive", "account manager", "business development",
            "bdm", "gsm", "dsm", "rsm", "isr", "tsr", "dsr", "rsr", "cmo",
            "realtor", "real estate", "leasing", "membership", "buyer",
            "purchaser", "purchasing", "procurement", "contracts",
            "contract specialist", "contract officer", "contracting",
            "negotiator", "distributor", "wholesaler", "auctioneer",
            "merchandiser", "merchandising", "merchandise", "pricing",
            "service advisor", "service writer", "auto dealer", "car dealer",
            "dealership", "parts", "counterman",
        ],
    },

    # --------------------------------------------------------
    # DISEÑO, MEDIOS, ARTE Y COMUNICACIÓN
    # --------------------------------------------------------
    "Design & Creative": {
        "exact": ["communication", "design", "model", "grip"],
        "specific": ["video production", "film production"],
        "keywords": [
            "designer", "graphic design", "graphics", "creative", "artist",
            "illustrator", "animator", "photographer", "videographer",
            "producer", "publisher", "editor", "proofreader", "writer",
            "copywriter", "author", "journalist", "photojournalist",
            "correspondent", "reporter", "anchor", "broadcaster", "dj",
            "disc jockey", "publicist", "public relations", "entertainer",
            "entertainment", "performer", "musician", "singer", "dancer",
            "actor", "referee", "stagehand", "set dresser",
            "content creator", "jeweler", "florist", "ceramist",
            "interpreter", "translator", "linguist",
            "communications specialist", "communications manager",
            "communications director",
        ],
    },

    # --------------------------------------------------------
    # BELLEZA, CUIDADO PERSONAL Y RECREACIÓN
    # --------------------------------------------------------
    "Personal Care & Beauty": {
        "specific": [
            "massage", "personal trainer", "fitness trainer",
            "fitness instructor", "yoga instructor", "makeup artist",
        ],
        "keywords": [
            "stylist", "hairstylist", "hairstyle", "hairdresser", "hair",
            "colorist", "cosmetologist", "cosmetology", "beautician",
            "barber", "esthetician", "aesthetician", "manicurist",
            "nail technician", "nail tech", "lmt", "nanny", "babysitter",
            "personal care", "personal assistant", "groomer", "dog walker",
            "pet sitter", "zookeeper", "golf", "tennis", "mortician",
            "funeral",
        ],
    },

    # --------------------------------------------------------
    # HOSTELERÍA, ALIMENTACIÓN Y CASINOS
    # --------------------------------------------------------
    "Hospitality & Food Service": {
        "exact": ["dealer", "crew", "runner", "prep", "steward", "decorator"],
        "specific": ["night auditor", "night audit", "food production", "food safety"],
        "keywords": [
            "table game", "casino", "blackjack", "poker", "card dealer",
            "games dealer", "21 dealer", "dual rate", "slot", "server",
            "waiter", "waitress", "wait staff", "waitstaff", "food runner",
            "barback", "bartender", "barista", "sommelier", "wine steward",
            "stewarding", "host", "hostess", "busser", "crew member", "chef",
            "cook", "cooker", "cooking", "culinary", "kitchen", "pantry",
            "dishwasher", "baker", "bakery", "pastry", "cake", "deli",
            "butcher", "meat", "meatcutter", "catering", "caterer",
            "banquet", "food and beverage", "food service", "food server",
            "food handler", "food prep", "food preparation", "fountain",
            "dietary", "cafeteria", "nutrition service", "child nutrition",
            "restaurant", "hotel", "hospitality", "housekeeper",
            "housekeeping", "houseman", "houseperson", "maid", "butler",
            "laundry", "room service", "concierge", "bellman", "valet",
            "usher", "reservation", "reservationist", "tour guide",
            "travel agent", "travel advisor", "event planner",
            "wedding planner", "meeting planner",
        ],
        "generic": ["attendant"],
    },

    # --------------------------------------------------------
    # ADMINISTRACIÓN
    # --------------------------------------------------------
    "Administration": {
        "exact": ["office", "support", "aa", "ea", "oa", "adm"],
        "keywords": [
            "admin", "administration", "administrative", "secretary",
            "receptionist", "reception", "clerical", "typist",
            "stenographer", "data entry", "order entry", "data processor",
            "data processing", "word processor", "document control",
            "document processor", "records", "registration", "scheduling",
            "front office", "business office", "office staff",
            "support staff", "office support", "office service",
            "business support", "program support", "support service",
            "office work", "office worker", "office assistance",
            "office manager", "office manage", "office coordinator",
            "office clerk", "file clerk", "executive assistant",
        ],
        "generic": ["assistant", "administrator"],
    },

    # --------------------------------------------------------
    # OPERACIONES (solo títulos exactos)
    # --------------------------------------------------------
    "Operations": {
        "exact": [
            "operation", "operations manager", "operations supervisor",
            "operations specialist", "operations coordinator",
            "operations director", "operations lead", "operations officer",
            "operations support", "operations planner", "business operations",
            "field operations", "planner", "scheduler", "master scheduler",
            "master planner", "project planner", "project scheduler",
            "planner/scheduler",
        ],
    },

    # --------------------------------------------------------
    # SERVICIOS PROFESIONALES
    # --------------------------------------------------------
    "Business & Professional Services": {
        "exact": ["professional", "staff", "employee", "self", "employed"],
        "specific": [
            "independent contractor", "government contractor",
            "defense contractor", "insurance adjuster",
            "claims representative", "claim representative",
            "insurance producer", "state farm",
        ],
        "keywords": [
            "consultant", "consulting", "partner", "broker", "insurance",
            "contract administrator", "self employed", "self-employed",
            "sole proprietor", "sole member", "freelancer", "freelance",
            "agency principal", "landman", "business advisor",
            "business officer", "project officer", "development officer",
            "major gift", "fundraiser", "fundraising", "strategist",
            "subject matter expert", "sme", "professional staff",
        ],
        "generic": ["specialist", "associate", "agent", "representative", "expert"],
    },

    # --------------------------------------------------------
    # DIRECCIÓN Y LIDERAZGO
    # --------------------------------------------------------
    "Management & Executive": {
        "exact": ["member", "pm", "dm", "rm", "cos", "regional", "superior", "manage"],
        "specific": ["deputy director", "production lead"],
        "keywords": [
            "ceo", "coo", "cao", "cco", "cro", "cso", "president", "vp",
            "svp", "avp", "evp", "rvp", "chief", "chairman", "boss",
            "executive", "owner", "proprietor", "shareholder", "franchisee",
            "landlord", "founder", "cofounder", "managing member",
            "board member", "managing principal", "gm", "agm", "asm", "rgm",
            "apm", "pmo", "general manager", "superintendent",
            "department head", "division head", "head of", "team leader",
            "assistant manager", "assistant store manager",
            "assistant director", "associate director", "associate manager",
            "lead technician", "lead operator", "lead specialist",
        ],
        "generic": [
            "manager", "management", "director", "supervisor", "lead",
            "leader", "head",
        ],
    },
}


# ============================================================
# 4. AGRUPACIÓN FINAL DE SECTORES
# Las reglas clasifican con el detalle de arriba y después
# estos sectores se juntan para tener grupos más grandes.
# Los sectores que no aparecen aquí conservan su nombre.
# ============================================================

SECTOR_GROUPS = {
    "Other Occupations": "Other / Inferred Profile",
    "Inferred Profile": "Other / Inferred Profile",

    "Business & Analytics": "Business & Professional Services",
    "Business & Professional Services": "Business & Professional Services",

    "Administration": "Administration & HR",
    "Human Resources & Coordination": "Administration & HR",

    "Transportation": "Transportation & Logistics",
    "Logistics & Warehousing": "Transportation & Logistics",

    "Manufacturing & Production": "Industrial & Skilled Trades",
    "Facilities & Skilled Trades": "Industrial & Skilled Trades",

    "Public Safety": "Public Safety & Military",
    "Military": "Public Safety & Military",

    "Healthcare": "Healthcare & Personal Care",
    "Personal Care & Beauty": "Healthcare & Personal Care",

    "Government & Public Administration": "Government & Legal",
    "Legal": "Government & Legal",
}


# ============================================================
# 5. CONSTRUIR LAS REGLAS EN ORDEN DE PRIORIDAD
# ============================================================

TIERS = ["exact", "specific", "keywords", "generic"]


def build_pattern(terms, exact=False):
    terms = sorted(set(terms), key=len, reverse=True)
    # Términos de 4+ letras aceptan plural (-s / -es);
    # las siglas cortas (rn, pa, co...) no, para evitar choques
    long_terms = "|".join(re.escape(t) for t in terms if len(t) >= 4)
    short_terms = "|".join(re.escape(t) for t in terms if len(t) < 4)
    parts = []
    if long_terms:
        parts.append(f"(?:{long_terms})(?:s|es)?")
    if short_terms:
        parts.append(f"(?:{short_terms})")
    body = "|".join(parts)
    return rf"^(?:{body})$" if exact else rf"\b(?:{body})\b"


RULES = [
    (tier, sector, build_pattern(levels[tier], exact=(tier == "exact")))
    for tier in TIERS
    for sector, levels in SECTORS.items()
    if levels.get(tier)
]


def find_repeated_terms():
    """Términos que aparecen en más de un sector o nivel."""
    term_locations = {}
    for sector, levels in SECTORS.items():
        for tier, terms in levels.items():
            for term in terms:
                term_locations.setdefault(term, []).append(f"{sector} [{tier}]")
    return {t: locs for t, locs in term_locations.items() if len(locs) > 1}


# ============================================================
# 6. CLASIFICACIÓN
# Se clasifica cada título único una sola vez y luego se mapea.
#   employment_sector -> sector agrupado (el que se usa)
#   sector_detail     -> sector antes de agrupar
# ============================================================


def classify_titles(titles, detail=False):
    """Clasifica una Serie de títulos.

    Con detail=False devuelve una Serie con el sector agrupado.
    Con detail=True devuelve un DataFrame con employment_sector,
    sector_detail, match_tier y normalized_title.
    """
    raw = titles.astype(object).fillna("").astype(str)
    unique_raw = pd.Series(raw.unique(), dtype=object)
    normalized = normalize_titles(unique_raw)

    sector = pd.Series(None, index=unique_raw.index, dtype=object)
    tier = pd.Series(None, index=unique_raw.index, dtype=object)

    # Cada título se queda con la primera regla que coincide
    for tier_name, sector_name, pattern in RULES:
        pending = sector.isna()
        if not pending.any():
            break
        hits = normalized[pending].str.contains(pattern, regex=True, na=False)
        matched = hits.index[hits.to_numpy()]
        sector.loc[matched] = sector_name
        tier.loc[matched] = tier_name

    empty = sector.isna() & normalized.eq("")
    sector.loc[empty] = "Unknown Employment Profile"
    tier.loc[empty] = "empty"
    sector = sector.fillna("Other Occupations")
    tier = tier.fillna("no match")

    lookup = pd.DataFrame(
        {
            "employment_sector": sector.replace(SECTOR_GROUPS).to_numpy(),
            "sector_detail": sector.to_numpy(),
            "match_tier": tier.to_numpy(),
            "normalized_title": normalized.to_numpy(),
        },
        index=unique_raw.to_numpy(),
    )

    if not detail:
        return raw.map(lookup["employment_sector"]).rename("employment_sector")

    return pd.DataFrame(
        {column: raw.map(lookup[column]) for column in lookup.columns},
        index=titles.index,
    )


def build_employment_features(df, impute=True):
    """Imputa los títulos vacíos y añade la columna employment_sector.

    Devuelve una copia del DataFrame. Con impute=False solo clasifica
    (útil si el DataFrame no tiene emp_length, annual_income o
    earliest_cr_line_year).
    """
    df = impute_missing_titles(df) if impute else df.copy()
    df["employment_sector"] = classify_titles(df["emp_title"])
    return df


# ============================================================
# 7. REVISIÓN DEL RESULTADO
# ============================================================


def sector_distribution(df, column="employment_sector"):
    """Conteo y porcentaje de cada sector."""
    summary = (
        df[column]
        .value_counts()
        .rename_axis(column)
        .reset_index(name="count")
    )
    summary["percentage"] = (summary["count"] / len(df) * 100).round(2)
    return summary


def unmatched_titles(titles):
    """Títulos (normalizados) que no encajaron en ninguna regla."""
    detail = classify_titles(titles, detail=True)
    is_unmatched = detail["sector_detail"].eq("Other Occupations")
    counts = (
        detail.loc[is_unmatched, "normalized_title"]
        .value_counts()
        .rename_axis("normalized_title")
        .reset_index(name="count")
    )
    counts["cum_pct_of_other"] = (
        counts["count"].cumsum() / counts["count"].sum() * 100
    ).round(2)
    return counts


# Títulos revisados en cada lote, para comprobar que siguen
# clasificándose como se acordó
TITLES_TO_CHECK = [
    # Lote 1
    "Officer", "Forman", "Administrative", "Accounts Payable", "Lead",
    "Producer", "Lieutenant", "Specialist", "Correction Officer",
    "Claims Adjuster", "Librarian", "Medical Technologist", "Assistant",
    "Service Advisor", "Dealer", "Mail Carrier", "Lineman", "Nanny", "CPA",
    "Architect", "Scientist", "Assembly", "Carrier", "Contract Specialist",
    "Baker", "Medical Coder", "Faculty", "Editor",

    # Lote 2
    "IT", "Loan Processor", "Contractor", "Housekeeping", "SVP",
    "Psychologist", "School Psychologist", "Planner", "Paraprofessional",
    "CTO", "Advisor", "HR", "HR Generalist", "Processor", "Pipefitter",
    "Porter", "AVP", "Stocker", "Quality Control",
    "Speech Language Pathologist", "Purchasing", "Postmaster", "Janitor",
    "Table Games Dealer",

    # Lote 3
    "Manger", "Managing Member", "Registrar", "Cleaner", "Journeyman",
    "PCA", "Mailhandler", "Care Giver", "Office", "Biller", "E6",
    "Pressman", "PTA", "Appraiser", "Doorman", "Dispatch",
    "Home Health Aid", "Contracting Officer", "Founder", "Researcher",
    "Driller", "Trader", "Examiner", "Member", "E5", "PCT",
    "Warehouseman", "Closer",

    # Lote 4
    "Receiver", "TSO", "Comptroller", "Sole Proprietor", "Finisher",
    "Major", "CSA", "Coder", "Group Leader", "Meat Cutter", "Detailer",
    "Butcher", "Psychotherapist", "Quality", "Drafter", "RDA",
    "Crew leader", "Patrolman", "Machinest", "Superintendant",
    "Superviser", "RCA", "Interpreter", "Professional", "QA", "MA",
    "Provider", "Reception",

    # Lote 5
    "COTA", "Locksmith", "Leadman", "Longshoreman", "Service",
    "Truckdriver", "Selector", "Apprentice", "Operations Officer",
    "Carman", "Shipper", "Staff", "Crew Member", "Builder", "Helper",
    "Orderfiller", "Federal Officer", "Senior Advisor", "Radiographer",
    "truckdriver", "Glazier", "Utility", "Printer", "Surveyor", "Deli",
    "Electrican", "Order Selector", "Communications",

    # Lote 6
    "Sorter", "Corporal", "CST", "Worker", "Medic", "Patent Examiner",
    "Commander", "Caretaker", "Commercial Lender", "Employee", "Rigger",
    "PSR", "Revenue Officer", "CBP Officer", "Expeditor", "PM", "ASM",
    "maintance", "Aide", "Master Scheduler", "Crew", "Technical Advisor",
    "AGM", "Pediatrician", "PA", "Economist", "Managment", "Maintance",

    # Lote 7
    "Pres", "Safety", "Purchaser", "SFC", "Tankerman", "ARNP", "CPhT",
    "NP", "Houseman", "Communications Officer", "Postdoctoral Fellow",
    "Facilitator", "Farmer", "Fitter", "RDH", "Transporter", "Mate",
    "Support Staff", "Eligibility Worker", "Transcriptionist",
    "Educational Diagnostician", "Safety Officer", "Insulator",
    "Wealth Advisor", "APRN", "Deportation Officer", "SSG",
    "Epidemiologist",

    # Lote 8 (muestra del CSV: errores, plurales y casos nuevos)
    "House Keeping", "Package Handler", "SGT", "Home Attendant",
    "Cake Decorator", "Book Keeper", "Technition", "Maneger", "Operater",
    "Supervisior", "Adminstrator", "Formen", "Salesmen", "Analysts",
    "Full Time", "Sever", "Owener", "GS-12", "Program Officer", "Miner",
    "Tool Maker", "Room Attendant", "Truck Diver", "Lt Col", "Acct Exec",
    "Para Professional", "Seamstress", "Mental Health Worker",
    "Specimen Processor", "Food Service Worker",
]


def check_titles():
    """Tabla con la clasificación de TITLES_TO_CHECK."""
    check = pd.DataFrame({"emp_title": TITLES_TO_CHECK}).drop_duplicates()
    return check.join(classify_titles(check["emp_title"], detail=True))


# ============================================================
# 8. USO DESDE LA TERMINAL
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Clasifica emp_title por sector y guarda el resultado."
    )
    parser.add_argument("input", nargs="?", help="CSV de entrada con la columna emp_title")
    parser.add_argument("output", nargs="?", help="CSV de salida con employment_sector")
    parser.add_argument("--no-impute", action="store_true",
                        help="No rellenar emp_title / emp_length vacíos")
    parser.add_argument("--check", action="store_true",
                        help="Mostrar la clasificación de los títulos de los lotes")
    args = parser.parse_args()

    repeated = find_repeated_terms()
    if repeated:
        print("TÉRMINOS REPETIDOS:")
        for term, locs in repeated.items():
            print(f"  {term!r}: {', '.join(locs)}")

    if args.check:
        pd.set_option("display.width", 200)
        print(check_titles().to_string(index=False))
        return

    if not args.input or not args.output:
        parser.error("indica el CSV de entrada y el de salida, o usa --check")

    df = pd.read_csv(args.input, low_memory=False)
    df = build_employment_features(df, impute=not args.no_impute)
    df.to_csv(args.output, index=False)

    print(sector_distribution(df).to_string(index=False))
    print(f"\nGuardado en {args.output}")


if __name__ == "__main__":
    main()
