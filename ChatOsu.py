# -*- coding: utf-8 -*-
"""
Chatbot osu! standard players (2007 -> today)

- Lit les données depuis un CSV (Kaggle ou local dans le repo)
- Chat Streamlit
- Utilise pandas pour interroger le dataset
- Utilise Groq (llama3-70b-8192) pour formuler les réponses en langage naturel
"""

import os
import re
import textwrap

import pandas as pd
import streamlit as st

try:
    from groq import Groq, BadRequestError
except ImportError:
    Groq = None
    BadRequestError = Exception


# ================== CONFIG CSV ==================

CANDIDATE_PATHS = [
    # Chemin Kaggle
    "/kaggle/input/osustandard-players-from-2007-to-today/2025_05_16_players_info.csv",
    # Racine du repo
    "2025_05_16_players_info.csv",
    # Optionnel: sous-dossier data
    "data/2025_05_16_players_info.csv",
]

# ================== CONFIG IA (GROQ) ==================

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama3-70b-8192"


def get_groq_client():
    if not GROQ_API_KEY or Groq is None:
        return None
    return Groq(api_key=GROQ_API_KEY)


groq_client = get_groq_client()

# ================== CHARGEMENT DES DONNÉES ==================


@st.cache_data(show_spinner=True)
def load_players():
    """
    Charge le CSV des joueurs de manière robuste.
    Retourne (df, colmap, parse_report).
    """
    csv_path = None
    for path in CANDIDATE_PATHS:
        if os.path.exists(path):
            csv_path = path
            break

    if csv_path is None:
        raise FileNotFoundError(
            "Impossible de trouver le CSV.\n"
            "Chemins testés :\n- " + "\n- ".join(CANDIDATE_PATHS)
        )

    parse_attempts = []

    # 1) Tentative standard
    df = None
    try:
        df = pd.read_csv(csv_path)
        parse_attempts.append("pandas default ✓")
    except Exception as e:
        parse_attempts.append(f"pandas default ✗ ({e})")

    # 2) Sniffer (auto-détection séparateur)
    if df is None:
        try:
            import csv

            with open(csv_path, "r", encoding="utf-8") as f:
                sample = f.read(50000)
            dialect = csv.Sniffer().sniff(sample)
            df = pd.read_csv(csv_path, sep=dialect.delimiter)
            parse_attempts.append(f"Sniffer autodetect ✓ sep='{dialect.delimiter}'")
        except Exception as e:
            parse_attempts.append(f"Sniffer autodetect ✗ ({e})")

    # 3) séparateur ';'
    if df is None:
        try:
            df = pd.read_csv(csv_path, sep=";")
            parse_attempts.append("sep=';' ✓")
        except Exception as e:
            parse_attempts.append(f"sep=';' ✗ ({e})")

    # 4) engine='python', on_bad_lines='skip'
    if df is None:
        try:
            df = pd.read_csv(csv_path, engine="python", sep=None, on_bad_lines="skip")
            parse_attempts.append("engine='python' autodetect ✓")
        except Exception as e:
            parse_attempts.append(f"engine='python' ✗ ({e})")

    # 5) encoding latin1
    if df is None:
        try:
            df = pd.read_csv(csv_path, encoding="latin1")
            parse_attempts.append("encoding='latin1' ✓")
        except Exception as e:
            parse_attempts.append(f"latin1 ✗ ({e})")

    if df is None:
        st.error(
            "**Impossible de parser le CSV**, même après plusieurs tentatives.\n\n"
            "Rapport :\n- " + "\n- ".join(parse_attempts)
        )
        return pd.DataFrame(), {}, "\n".join(parse_attempts)

    # Détection des colonnes utiles
    cols = {c.lower(): c for c in df.columns}

    username_col = cols.get("username") or cols.get("user") or cols.get("name")
    country_col = cols.get("country")
    pp_col = cols.get("pp")
    rank_col = cols.get("global ranking") or cols.get("global_rank") or cols.get("rank")

    colmap = {
        "username": username_col,  # 'username'
        "country": country_col,  # 'Country'
        "pp": pp_col,  # 'pp'
        "rank": rank_col,  # 'Global Ranking'
    }

    return df, colmap, "\n".join(parse_attempts)


# ================== COEUR "PANDAS" : RÉPONSE STRUCTURÉE ==================


def structured_answer(question: str, df: pd.DataFrame, colmap: dict) -> tuple[str, str]:
    """
    Renvoie (base_answer, context) :
      - base_answer : texte brut (sans IA)
      - context : données détaillées pour l'IA (liste de joueurs, stats, etc.)
    """
    if df.empty:
        return "Je n'ai pas pu charger les données des joueurs.", ""

    q = question.lower()
    username_col = colmap.get("username")
    country_col = colmap.get("country")
    pp_col = colmap.get("pp")
    rank_col = colmap.get("rank")

    # 0) Aide
    if "help" in q or "aide" in q:
        txt = (
            "Je peux t'aider à explorer les joueurs osu! standard.\n\n"
            "Exemples :\n"
            "- Combien de joueurs y a-t-il ?\n"
            "- Top 10 joueurs par pp\n"
            "- Infos sur le joueur Cookiezi\n"
            "- Combien de joueurs du FR ?\n"
            "- Qui est le joueur avec le plus de pp ?\n"
        )
        return txt, ""

    # 1) Colonnes
    if "colonnes" in q or "columns" in q:
        cols_txt = "\n".join(f"- {c}" for c in df.columns)
        return f"Voici les colonnes du dataset :\n\n{cols_txt}", cols_txt

    # 2) Nombre de joueurs
    if ("combien" in q or "how many" in q) and any(
        x in q for x in ["joueur", "players", "player"]
    ):
        txt = f"Le dataset contient {len(df):,} joueurs.".replace(",", " ")
        return txt, txt

    # 3) "Qui est <nom>"
    if ("qui est " in q or "who is " in q) and username_col:
        lower = question.lower()
        if "qui est " in lower:
            start = lower.find("qui est ") + len("qui est ")
        else:
            start = lower.find("who is ") + len("who is ")
        name = question[start:].strip(" ?!.\"'")
        if not name:
            return "Tu peux me demander : `Qui est Cookiezi ?`", ""

        sub = df[df[username_col].astype(str).str.contains(name, case=False, na=False)]
        if sub.empty:
            return f"Aucun joueur ne correspond à '{name}'.", ""

        sub = sub.head(3)
        lines = []
        for _, row in sub.iterrows():
            base = f"{row[username_col]}"
            if country_col:
                base += f" ({row[country_col]})"
            extras = []
            if pp_col and pd.notna(row[pp_col]):
                extras.append(f"pp={row[pp_col]}")
            if rank_col and pd.notna(row[rank_col]):
                extras.append(f"global_rank={row[rank_col]}")
            if extras:
                base += " | " + ", ".join(extras)
            lines.append(base)

        context = "\n".join(lines)
        return f"J'ai trouvé {len(sub)} joueur(s) correspondant à '{name}'.", context

    # 4) Top N / joueur avec le plus de pp
    top_match = re.search(r"top\s+(\d+)", q)
    n = None
    if top_match:
        n = int(top_match.group(1))
    if n is None and ("plus de pp" in q or "most pp" in q or "le plus de pp" in q):
        n = 1

    if n is not None:
        metric = None
        if "pp" in q and pp_col:
            metric = pp_col
        elif "rank" in q and rank_col:
            metric = rank_col
        elif pp_col:
            metric = pp_col

        if metric is None:
            txt = (
                "Je ne trouve pas de colonne de pp ou de rank dans le dataset.\n"
                f"Colonnes disponibles : {', '.join(df.columns)}"
            )
            return txt, txt

        subset = df.dropna(subset=[metric])
        ascending = "rank" in metric.lower() or "global" in metric.lower()
        top_df = subset.sort_values(metric, ascending=ascending).head(n)

        lines = []
        for _, row in top_df.iterrows():
            base = f"{row[username_col]}" if username_col else "?"
            if country_col:
                base += f" ({row[country_col]})"
            base += f" | {metric}={row[metric]}"
            lines.append(base)

        context = "\n".join(lines)
        if n == 1:
            return f"Voici le meilleur joueur selon {metric}.", context
        else:
            return f"Voici le top {n} joueurs selon {metric}.", context

    # 5) Joueurs d'un pays
    if "pays" in q or "country" in q:
        if not country_col:
            txt = "Je ne trouve pas de colonne 'Country' dans le dataset."
            return txt, txt
        m = re.search(r"\b([A-Z]{2})\b", question)
        if m:
            code = m.group(1)
            sub = df[df[country_col] == code]
            if sub.empty:
                txt = f"Aucun joueur trouvé pour le pays {code}."
                return txt, txt
            txt = f"Il y a {len(sub):,} joueurs pour le pays {code}.".replace(",", " ")
            return txt, txt
        else:
            txt = (
                "Tu peux préciser un code pays en majuscules (FR, JP, US...).\n"
                "Exemple : `Combien de joueurs du FR ?`"
            )
            return txt, txt

    # 6) Infos sur "<joueur>" sans "qui est"
    if any(x in q for x in ["joueur", "player", "username", "user"]) and username_col:
        # on essaye de récupérer un pseudo entre guillemets ou dernier mot
        m = re.search(r'"([^"]+)"', question)
        if m:
            name = m.group(1)
        else:
            name = question.strip(" ?!.").split()[-1]

        sub = df[df[username_col].astype(str).str.contains(name, case=False, na=False)]
        if sub.empty:
            return f"Aucun joueur ne correspond à '{name}'.", ""

        sub = sub.head(3)
        lines = []
        for _, row in sub.iterrows():
            base = f"{row[username_col]}"
            if country_col:
                base += f" ({row[country_col]})"
            extras = []
            if pp_col and pd.notna(row[pp_col]):
                extras.append(f"pp={row[pp_col]}")
            if rank_col and pd.notna(row[rank_col]):
                extras.append(f"global_rank={row[rank_col]}")
            if extras:
                base += " | " + ", ".join(extras)
            lines.append(base)

        context = "\n".join(lines)
        return f"Voici quelques infos sur '{name}'.", context

    # 7) Fallback : quelques stats globales
    num_cols = df.select_dtypes("number").columns
    summary_lines = [
        f"Nombre de joueurs : {len(df):,}".replace(",", " "),
    ]
    for col in ["pp", "Level", "Medals"]:
        if col in df.columns:
            summary_lines.append(
                f"{col} : min={df[col].min():.2f}, max={df[col].max():.2f}, moyenne={df[col].mean():.2f}"
            )

    base = (
        "Je ne suis pas sûr de comprendre la question, "
        "mais voici quelques statistiques globales."
    )
    context = "\n".join(summary_lines)
    return base, context


# ================== COUCHE IA (GROQ) ==================


def call_llm(question: str, base_answer: str, context: str, history: list[dict]) -> str:
    """
    Appelle Groq (llama3-70b-8192) pour formuler une réponse finale.
    Si Groq n'est pas disponible, renvoie simplement base_answer + context.
    """
    if groq_client is None:
        # Pas de clé / pas de lib -> fallback
        if context:
            return f"{base_answer}\n\n{context}"
        return base_answer

    # On limite l'historique envoyé au modèle
    short_history = history[-6:]

    system_prompt = textwrap.dedent(
        """
        Tu es un assistant spécialisé dans le jeu osu! (mode standard).
        On te donne :
        - la question de l'utilisateur,
        - un début de réponse calculé avec des statistiques pandas,
        - un contexte contenant des lignes de joueurs (username, pays, pp, etc.).

        Ta mission :
        - Reformuler et compléter la réponse en français,
        - Expliquer clairement ce que signifient les chiffres si nécessaire,
        - Si le contexte ne contient pas l'information demandée, le dire honnêtement,
        - Ne jamais inventer de valeurs numériques qui ne sont pas dans le contexte.
        """
    ).strip()

    messages = [{"role": "system", "content": system_prompt}]

    for msg in short_history:
        messages.append(
            {
                "role": msg["role"],
                "content": msg["content"],
            }
        )

    user_content = f"Question : {question}\n\nRéponse de base : {base_answer}\n\nContexte (données brutes) :\n{context}"
    messages.append({"role": "user", "content": user_content})

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()
    except BadRequestError as e:
        return f"{base_answer}\n\n*(Erreur Groq: {e})*"
    except Exception as e:
        return f"{base_answer}\n\n*(Erreur inattendue Groq: {e})*"


# ================== APP STREAMLIT ==================


def main():
    st.set_page_config(page_title="Chatbot osu! players", page_icon="🎧")

    st.title("🎧 Chatbot osu! standard players (2007 → aujourd’hui)")
    st.write(
        "Je réponds à des questions sur les joueurs à partir du fichier CSV.\n"
        "Je combine pandas pour les stats et l'IA Groq (llama3-70b) pour formuler les réponses."
    )

    # Chargement des données
    try:
        df, colmap, parse_report = load_players()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Sidebar : infos
    st.sidebar.header("Données")
    st.sidebar.write(f"**Lignes :** {len(df):,}".replace(",", " "))
    st.sidebar.write(f"**Colonnes :** {len(df.columns)}")
    st.sidebar.write("Colonnes principales détectées :")
    for k, v in colmap.items():
        st.sidebar.write(f"- {k} → {v}")
    st.sidebar.markdown("---")
    st.sidebar.write("Méthode de parsing utilisée :")
    st.sidebar.code(parse_report or "pandas default")

    if groq_client is None:
        st.sidebar.warning("IA Groq non configurée (GROQ_API_KEY manquant ou lib 'groq' absente).")

    # Historique du chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Salut ! Je suis le bot osu! 🌟\n\n"
                    "Je peux t'aider à explorer les joueurs de ce dataset.\n"
                ),
            }
        ]

    # Affichage historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrée utilisateur
    if prompt := st.chat_input("Pose une question sur les joueurs osu!..."):
        # Ajouter la question à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Réponse structurée (pandas)
        base_answer, context = structured_answer(prompt, df, colmap)

        # Passage à l'IA Groq pour reformulation
        final_answer = call_llm(
            question=prompt,
            base_answer=base_answer,
            context=context,
            history=st.session_state.messages,
        )

        # Ajouter la réponse dans l'historique
        st.session_state.messages.append(
            {"role": "assistant", "content": final_answer}
        )

        st.rerun()


if __name__ == "__main__":
    main()
