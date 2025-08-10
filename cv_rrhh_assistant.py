import os, io, re, json, time
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ================== Lectores ==================
try:
    from PyPDF2 import PdfReader
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

try:
    import docx  # python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# ================== OpenAI ==================
OPENAI_API_KEY = "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" 
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None

# ================== Página ==================
st.set_page_config(page_title="Selector de CV con IA — NexInData", page_icon="🧠", layout="wide")
st.title("🧠 Selector de CV con IA — NexInData")
st.caption("MVP con GPT: pega el perfil del cargo, carga PDFs y obtén ranking + análisis visual por candidato.")

# ================== Utilidades ==================
EXPERIENCE_PATS = [
    r"(\d{1,2})\s*(años|anos)\s*(de\s*)?(experiencia)?",
    r"(\d{1,2})\+?\s*(years?)\s*(of\s*)?(experience)?",
]
PHONE_PAT = re.compile(r"(?:\+?56\s?)?(?:0?9\s?)?[\d\-\.\s]{8,12}")

LANG_MAP = {"A1":0.1,"A2":0.2,"B1":0.4,"B2":0.6,"C1":0.8,"C2":1.0,
            "BASICO":0.2,"INTERMEDIO":0.5,"AVANZADO":0.8,"FLUIDO":0.9,"NATIVO":1.0,
            "NO INDICA":0.0}

def extract_text_pdf(uploaded) -> str:
    name = uploaded.name.lower()
    if not name.endswith(".pdf") or not HAS_PYPDF2:
        return ""
    data = uploaded.read()
    buf = io.BytesIO(data)
    try:
        reader = PdfReader(buf)
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""

def clean_text(t: str) -> str:
    t = t or ""
    t = re.sub(r"\s+", " ", t)
    return t.strip()

# ================== GPT: extracción estructurada ==================
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "nombre": {"type": "string"},
        "telefono": {"type": "string"},
        "profesion": {"type": "string"},
        "universidad": {"type": "string"},
        "anios_experiencia": {"type": "integer"},
        "nivel_ingles": {"type": "string", "enum": ["A1","A2","B1","B2","C1","C2","BASICO","INTERMEDIO","AVANZADO","FLUIDO","NATIVO","NO INDICA"]},
        "habilidades": {"type": "object", "additionalProperties": {"type": "integer"}},
        "keywords_detectadas": {"type": "array", "items": {"type": "string"}},
        "score_idoneidad": {"type": "number"},
        "comentario": {"type": "string"}
    },
    "required": ["nombre","anios_experiencia","nivel_ingles","score_idoneidad"],
    "additionalProperties": False
}

SYSTEM_PROMPT = (
    "Eres un analista de RR.HH. Extrae del CV los campos solicitados y evalúa el ajuste al perfil. "
    "Responde estrictamente en JSON válido con el esquema dado. Si falta un dato, deja cadena vacía o un valor razonable."
)

USER_TEMPLATE = (
    "PERFIL DEL CARGO:\n{jd}\n\n"
    "CV (texto plano):\n{cv}\n\n"
    "Devuelve JSON con: nombre, telefono, profesion, universidad, anios_experiencia, nivel_ingles, "
    "habilidades (dict con conteo), keywords_detectadas (lista), score_idoneidad (0-1), comentario (breve)."
)

def gpt_extract(jd: str, cv_text: str) -> Dict[str, Any]:
    if not client:
        return {}
    content = USER_TEMPLATE.format(jd=jd[:6000], cv=cv_text[:6000])
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            temperature=0.2,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "cv_extraction",
                    "schema": EXTRACT_SCHEMA,
                    "strict": True
                }
            },
            max_tokens=500,
        )
        raw = resp.choices[0].message.content
        return json.loads(raw)
    except Exception:
        # Fallback a texto -> JSON
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + " Devuelve SOLO JSON."},
                {"role": "user", "content": content}
            ],
            temperature=0.2,
            max_tokens=700,
        )
        raw = resp.choices[0].message.content
        try:
            start = raw.find("{"); end = raw.rfind("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            return {}

# ================== UI: Paso 1 - Perfil ==================
st.subheader("1) Perfil del cargo")
jd_text = st.text_area(
    "Pega aquí la descripción del cargo",
    height=220,
    placeholder=(
        "Ej.: Buscamos Data Analyst con experiencia en Python, SQL y dashboards (Tableau/Power BI). "
        "Inglés B2+ y trato con stakeholders."
    ),
)

# ================== UI: Paso 2 - Botón para cargar PDFs ==================
st.subheader("2) Cargar CVs (PDF)")
files = []
if jd_text:
    col_btn = st.container()
    with col_btn:
        if st.button("📎 Cargar archivos (PDF)"):
            st.session_state["show_uploader"] = True
    if st.session_state.get("show_uploader", False):
        files = st.file_uploader(
            "Selecciona uno o varios PDF (hasta 200 archivos)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Puedes soltar una carpeta completa desde el Finder/Explorador.",
        ) or []
else:
    st.info("Primero ingresa la descripción del cargo. Luego podrás cargar los PDFs.")

# Límite de archivos
if files and len(files) > 200:
    st.error(f"Seleccionaste {len(files)} archivos. El máximo permitido es 200.")
    files = files[:200]

if files:
    st.success(f"{len(files)} archivos listos para procesar.")

st.markdown("---")

# ================== Proceso ==================
if jd_text and files:
    rows: List[Dict[str, Any]] = []
    progress = st.progress(0)
    status = st.empty()

    for i, up in enumerate(files, start=1):
        status.text(f"Procesando {i}/{len(files)}: {up.name}")
        raw = extract_text_pdf(up)
        up.seek(0)
        txt = clean_text(raw)

        result: Dict[str, Any] = gpt_extract(jd_text, txt) if client and OPENAI_API_KEY else {}

        # Fallback básico si la API falla
        if not result:
            # años de experiencia heurístico
            years = 0
            for pat in EXPERIENCE_PATS:
                for m in re.finditer(pat, txt.lower()):
                    try:
                        years = max(years, int(m.group(1)))
                    except Exception:
                        pass
            phone = PHONE_PAT.search(txt)
            result = {
                "nombre": re.sub(r"\.pdf$", "", up.name, flags=re.I),
                "telefono": phone.group(0).strip() if phone else "",
                "profesion": "",
                "universidad": "",
                "anios_experiencia": years,
                "nivel_ingles": "NO INDICA",
                "habilidades": {},
                "keywords_detectadas": [],
                "score_idoneidad": 0.5,
                "comentario": "(Resumen heurístico por falta de API)"
            }

        # Normalizaciones
        result["anios_experiencia"] = int(result.get("anios_experiencia") or 0)
        try:
            result["score_idoneidad"] = float(result.get("score_idoneidad") or 0.0)
        except Exception:
            result["score_idoneidad"] = 0.0
        if isinstance(result.get("habilidades"), list):
            result["habilidades"] = {h: 1 for h in result["habilidades"]}

        row = {
            "archivo": up.name,
            **result,
            "texto_cv": txt,
        }
        rows.append(row)

        progress.progress(i / len(files))
        # Pequeño respiro para UI
        time.sleep(0.01)

    status.empty()
    progress.empty()

    df = pd.DataFrame(rows).sort_values("score_idoneidad", ascending=False).reset_index(drop=True)

    st.subheader("3) Ranking de candidatos")
    st.dataframe(df[[
        "archivo","nombre","telefono","profesion","universidad",
        "anios_experiencia","nivel_ingles","score_idoneidad"
    ]], use_container_width=True)

    csv = df.drop(columns=["texto_cv"]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar resultados (CSV)", csv, file_name="ranking_candidatos.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("4) Detalle por candidato + visualizaciones")
    choice = st.selectbox("Selecciona un candidato", df["nombre"].tolist())
    row = df[df["nombre"] == choice].iloc[0]

    c1,c2,c3 = st.columns([1,1,1])
    with c1:
        st.metric("Score de idoneidad", f"{row['score_idoneidad']:.2f}")
        st.metric("Años de experiencia", int(row["anios_experiencia"]))
    with c2:
        st.metric("Nivel de inglés", row["nivel_ingles"])
        st.metric("CV fuente", row["archivo"])
    with c3:
        st.write(":memo: Comentario")
        st.write(row.get("comentario",""))

    # ---- Gauge score ----
    fig_score = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(row["score_idoneidad"]) * 100,
        gauge={"axis": {"range": [0,100]}, "bar": {"thickness": 0.6}},
        title={"text": "Score (0–100)"},
    ))
    st.plotly_chart(fig_score, use_container_width=True)

    # ---- Barras: habilidades ----
    skills: Dict[str,int] = row.get("habilidades") or {}
    if isinstance(skills, dict) and len(skills) > 0:
        df_sk = pd.DataFrame({"skill": list(skills.keys()), "conteo": list(skills.values())}) \
                .sort_values("conteo", ascending=False)
        fig_sk = px.bar(df_sk, x="skill", y="conteo", title="Habilidades detectadas (frecuencia)")
        fig_sk.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_sk, use_container_width=True)
    else:
        st.info("No se detectaron habilidades en el JSON. Ajusta el perfil o el CV.")

    # ---- Radar: Experiencia vs Inglés ----
    lvl = str(row["nivel_ingles"]).upper()
    en_val = LANG_MAP.get(lvl, 0.0)
    radar_df = pd.DataFrame({
        "eje": ["Experiencia","Inglés"],
        "valor": [min(int(row["anios_experiencia"]) / 10.0, 1.0), en_val]
    })
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=radar_df["valor"], theta=radar_df["eje"], fill='toself', name=choice))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), title="Perfil (normalizado)")
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Ingresa el perfil y luego carga los PDFs para procesar.")
