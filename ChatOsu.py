# -*- coding: utf-8 -*-
"""
Chatbot osu! standard players (2007 -> today)

- Lit les données depuis un CSV (Kaggle ou local dans le repo)
- Interface de chat avec Streamlit
- Répond à des questions simples sur les joueurs / pays / top rankings
"""

import os
import re

import pandas as pd
import streamlit as st


# ================== CHEMINS POSSIBLES VERS LE CSV ==================

CANDIDATE_PATHS = [
    # Chemin Kaggle (quand tu lances sur Kaggle)
    "/kaggle/input/osustandard-players-from-2007-to-today/2025_05_16_players_info.csv",
    # Chemin local racine du repo
    "2025_05_16_players_info.csv",
]


# ================== CHARGEMENT DES DONNÉES ==================

@st.cache_data(show_spinner=True)
@st.cache_data(show_spinner=True)
@st.cache_data(show_spinner=True)
def load_players():
    """
    Charge le CSV des joueurs de manière extrêmement robuste.
    Essaie plusieurs parseurs, séparateurs et encodings.
    Retourne (df, colmap).
    """

    # 1. Vérifier les chemins possibles
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

    # 2. Essais successifs de parsing
    parse_attempts = []

    # ---- Tentative 1 : pandas standard (par défaut)
    try:
        df = pd.read_csv(csv_path)
        parse_attempts.append("pandas default ✓")
    except Exception as e:
        parse_attempts.append(f"pandas default ✗ ({e})")
        df = None

    # ---- Tentative 2 : autodétection du séparateur avec Sniffer
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
            df = None

    # ---- Tentative 3 : séparateur point-virgule
    if df is None:
        try:
            df = pd.read_csv(csv_path, sep=";")
            parse_attempts.append("sep=';' ✓")
        except Exception as e:
            parse_attempts.append(f"sep=';' ✗ ({e})")
            df = None

    # ---- Tentative 4 : parser python + skip bad lines
    if df is None:
        try:
            df = pd.read_csv(
                csv_path,
                engine="python",
                sep=None,
                on_bad_lines="skip",
            )
            parse_attempts.append("engine='python' autodetect ✓")
        except Exception as e:
            parse_attempts.append(f"engine='python' ✗ ({e})")
            df = None

    # ---- Tentative 5 : encoding latin1
    if df is None:
        try:
            df = pd.read_csv(csv_path, encoding="latin1")
            parse_attempts.append("encoding='latin1' ✓")
        except Exception as e:
            parse_attempts.append(f"latin1 ✗ ({e})")
            df = None

    # ---- Si toujours aucun succès
    if df is None:
        st.error(
            "**Impossible de parser le CSV**, même après plusieurs tentatives.\n\n"
            "Rapport :\n- " + "\n- ".join(parse_attempts)
        )
        return pd.DataFrame(), {}

    # On affiche le rapport dans la console Streamlit
    st.sidebar.info(
        "Méthode de parsing utilisée :\n\n- " + "\n- ".join(parse_attempts)
    )

    # =============================
    # Détection des colonnes utiles
    # =============================
    cols = {c.lower(): c for c in df.columns}

    username_col = cols.get("username") or cols.get("user") or cols.get("name")
    country_col = cols.get("country") or cols.get("country_code")
    pp_col = cols.get("pp") or cols.get("performance_points")
    rank_col = cols.get("global_rank") or cols.get("rank")

    colmap = {
        "username": username_col,
        "country": country_col,
        "pp": pp_col,
        "rank": rank_col,
    }

    return df, colmap




# ================== LOGIQUE DU CHATBOT ==================

def answer_question(question: str, df: pd.DataFrame, colmap: dict) -> str:
    """
    Génère une réponse à partir de la question et du DataFrame.
    Pas d'IA externe : juste des règles + pandas.
    """
    if df.empty:
        return "Je n'ai pas pu charger les données des joueurs."

    q = question.lower()

    # --- 0. Aide / exemples ---
    if "help" in q or "aide" in q or "exemple" in q:
        return (
            "Je peux t'aider à explorer les joueurs osu! standard.\n\n"
            "Exemples de questions :\n"
            "- `Combien de joueurs y a-t-il au total ?`\n"
            "- `Montre-moi les colonnes disponibles`\n"
            "- `Top 10 joueurs par pp`\n"
            "- `Top 5 joueurs du JP par pp`\n"
            "- `Infos sur le joueur Cookiezi`\n"
            "- `Qui est peppy ?`\n"
            "- `Qui est le joueur avec le plus de pp ?`\n"
        )

    # --- 1. Colonnes / structure ---
    if "colonnes" in q or "columns" in q or "feature" in q:
        return "Les colonnes disponibles sont :\n\n- " + "\n- ".join(df.columns)

    # --- 2. Nombre total de joueurs ---
    if ("combien" in q or "how many" in q) and ("joueur" in q or "player" in q or "players" in q):
        return f"Le dataset contient **{len(df):,}** joueurs.".replace(",", " ")

    # --- 3. "Qui est <nom>" (ex: qui est peppy ?) ---
    if ("qui est " in q or "who is " in q) and colmap.get("username"):
        lower = question.lower()
        if "qui est " in lower:
            start = lower.find("qui est ") + len("qui est ")
        else:
            start = lower.find("who is ") + len("who is ")

        # On récupère ce qui suit "qui est"
        name = question[start:].strip(" ?!.\"'")

        if not name:
            return "Tu peux demander par ex. : `Qui est peppy ?` ou `Who is Cookiezi ?`"

        username_col = colmap["username"]
        sub = df[df[username_col].astype(str).str.contains(name, case=False, na=False)]

        if sub.empty:
            return f"Aucun joueur ne correspond à **{name}**."

        sub = sub.head(3)
        lines = []
        country_col = colmap.get("country")
        pp_col = colmap.get("pp")
        rank_col = colmap.get("rank")

        for _, row in sub.iterrows():
            base = f"**{row[username_col]}**"
            if country_col and country_col in row:
                base += f" ({row[country_col]})"

            extras = []
            if pp_col and pp_col in row and pd.notna(row[pp_col]):
                extras.append(f"{pp_col} = {row[pp_col]}")
            if rank_col and rank_col in row and pd.notna(row[rank_col]):
                extras.append(f"{rank_col} = {row[rank_col]}")

            if extras:
                base += " — " + ", ".join(extras)

            lines.append(base)

        return "Voici ce que j’ai trouvé :\n\n" + "\n".join(f"- {ln}" for ln in lines)

    # --- 4. Top N joueurs (global) ---
    import re
    top_match = re.search(r"top\s+(\d+)", q)
    n = None
    if top_match:
        n = int(top_match.group(1))
    # cas spécial : "joueur avec le plus de pp"
    if n is None and ("plus de pp" in q or "most pp" in q or "le plus de pp" in q):
        n = 1

    if n is not None:
        metric = None

        if "pp" in q and colmap.get("pp"):
            metric = colmap["pp"]
        elif "rank" in q and colmap.get("rank"):
            metric = colmap["rank"]
        elif colmap.get("pp"):
            metric = colmap["pp"]

        if metric is None:
            return (
                "Je ne trouve pas de colonne de pp ou de rank dans le dataset.\n\n"
                "Colonnes dispo : " + ", ".join(df.columns)
            )

        subset = df.dropna(subset=[metric])
        ascending = "rank" in metric.lower()  # pour un rang, plus petit = meilleur
        top_df = subset.sort_values(metric, ascending=ascending).head(n)

        username_col = colmap.get("username")
        country_col = colmap.get("country")

        lines = []
        for _, row in top_df.iterrows():
            name = str(row[username_col]) if username_col else "?"
            val = row[metric]
            country = str(row[country_col]) if country_col else "??"
            lines.append(f"{name} ({country}) — {metric} = {val}")

        if n == 1:
            return f"Le joueur avec le plus de **{metric}** est :\n\n- {lines[0]}"
        else:
            header = f"Top {n} joueurs selon **{metric}** :\n\n"
            return header + "\n".join(f"- {ln}" for ln in lines)

    # --- 5. Joueurs d'un pays ---
    if "pays" in q or "country" in q:
        country_col = colmap.get("country")
        if not country_col:
            return "Je ne trouve pas de colonne 'country' dans le dataset."

        import re as _re
        country_match = _re.search(r"\b([A-Z]{2})\b", question)
        if country_match:
            code = country_match.group(1)
            sub = df[df[country_col] == code]
            if sub.empty:
                return f"Aucun joueur trouvé pour le pays **{code}**."
            return (
                f"Il y a **{len(sub):,}** joueurs pour le pays **{code}**.\n"
                "Tu peux aussi demander : `Top 10 joueurs du FR par pp`."
            ).replace(",", " ")
        else:
            return (
                "Tu peux préciser un code pays en majuscules, par ex. :\n"
                "- `Combien de joueurs du FR ?`\n"
                "- `Top 5 joueurs du JP par pp`"
            )

    # --- 6. Infos sur un joueur (version avec mot 'joueur'/'player') ---
    if "joueur" in q or "player" in q or "user" in q or "username" in q:
        username_col = colmap.get("username")
        if not username_col:
            return (
                "Je ne trouve pas de colonne de nom de joueur (`username`).\n"
                "Colonnes dispo : " + ", ".join(df.columns)
            )

        import re as _re
        quoted = _re.search(r'"([^"]+)"', question)
        if quoted:
            name = quoted.group(1)
        else:
            name = None
            for key in ["joueur", "player", "username", "user"]:
                if key in q:
                    after = question.lower().split(key, 1)[1].strip()
                    if after:
                        name = after
                        break

        if not name:
            return (
                "Tu peux demander par ex. :\n"
                "- `Infos sur le joueur Cookiezi`\n"
                "- `Dis-moi quelque chose sur le player \"WhiteCat\"`"
            )

        sub = df[df[username_col].astype(str).str.contains(name, case=False, na=False)]
        if sub.empty:
            return f"Aucun joueur ne correspond à **{name}**."

        sub = sub.head(3)
        lines = []
        country_col = colmap.get("country")
        pp_col = colmap.get("pp")
        rank_col = colmap.get("rank")

        for _, row in sub.iterrows():
            base = f"**{row[username_col]}**"
            if country_col and country_col in row:
                base += f" ({row[country_col]})"

            extras = []
            if pp_col and pp_col in row and pd.notna(row[pp_col]):
                extras.append(f"{pp_col} = {row[pp_col]}")
            if rank_col and rank_col in row and pd.notna(row[rank_col]):
                extras.append(f"{rank_col} = {row[rank_col]}")

            if extras:
                base += " — " + ", ".join(extras)

            lines.append(base)

        return "Voici ce que j’ai trouvé :\n\n" + "\n".join(f"- {ln}" for ln in lines)

    # --- 7. Fallback : quelques stats globales ---
    num_cols = df.select_dtypes("number").columns
    text = [
        "Je ne suis pas sûr de comprendre ta question, mais voici quelques infos :",
        f"- Nombre de joueurs : **{len(df):,}**".replace(",", " "),
    ]
    for col in num_cols[:3]:
        text.append(
            f"- {col} : min={df[col].min():.2f}, max={df[col].max():.2f}, moyenne={df[col].mean():.2f}"
        )

    text.append("\nTu peux demander : `help` ou `aide` pour voir des exemples de questions.")
    return "\n".join(text)



# ================== APP STREAMLIT ==================

def main():
    st.set_page_config(page_title="Chatbot osu! players", page_icon="🎧")

    st.title("🎧 Chatbot osu! standard players (2007 → aujourd’hui)")
    st.write(
        "Je réponds à des questions sur les joueurs à partir du fichier CSV.\n"
        "Pose ta question en bas. Tu peux taper `help` pour voir des exemples."
    )

    # Chargement des données
    try:
        df, colmap = load_players()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Résumé côté barre latérale
    st.sidebar.header("Données")
    st.sidebar.write(f"**Lignes :** {len(df):,}".replace(",", " "))
    st.sidebar.write(f"**Colonnes :** {len(df.columns)}")
    st.sidebar.write("Colonnes principales détectées :")
    for k, v in colmap.items():
        st.sidebar.write(f"- {k} → {v}")
    st.sidebar.caption("Si un nom de colonne ne correspond pas, adapte la fonction load_players().")

    # Historique du chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Salut ! Je suis le bot osu! 🌟\n\n"
                    "Je peux t'aider à explorer les joueurs de ce dataset.\n"
                    "Essaie par exemple :\n"
                    "- `Combien de joueurs y a-t-il ?`\n"
                    "- `Top 10 joueurs par pp`\n"
                    "- `Infos sur le joueur Cookiezi`\n"
                    "- `Combien de joueurs du FR ?`\n"
                ),
            }
        ]

    # Afficher l’historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrée utilisateur
    if prompt := st.chat_input("Pose une question sur les joueurs osu!..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        answer = answer_question(prompt, df, colmap)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        st.rerun()


if __name__ == "__main__":
    main()
