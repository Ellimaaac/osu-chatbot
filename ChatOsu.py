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
def load_players():
    """
    Charge le CSV des joueurs.

    Retourne : (df, colmap)
      - df : DataFrame complet
      - colmap : dict avec les colonnes importantes détectées
                 {"username": ..., "country": ..., "pp": ..., "rank": ...}
    """
    # 1) Trouver un chemin existant
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

    # 2) Charger le CSV
    df = pd.read_csv(csv_path)

    # 3) Détection des colonnes importantes
    cols = {c.lower(): c for c in df.columns}  # lowercase -> original

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
    Génère une réponse simple à partir de la question et du DataFrame.
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
            "- `Top 5 joueurs du Japon par pp`\n"
            "- `Infos sur le joueur Cookiezi`\n"
        )

    # --- 1. Colonnes / structure ---
    if "colonnes" in q or "columns" in q or "feature" in q:
        return "Les colonnes disponibles sont :\n\n- " + "\n- ".join(df.columns)

    # --- 2. Nombre total de joueurs ---
    if ("combien" in q or "how many" in q) and ("joueur" in q or "player" in q or "players" in q):
        return f"Le dataset contient **{len(df):,}** joueurs.".replace(",", " ")

    # --- 3. Top N joueurs (global) ---
    top_match = re.search(r"top\s+(\d+)", q)
    if top_match:
        n = int(top_match.group(1))
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
        ascending = "rank" in metric.lower()  # rank : plus petit = meilleur
        top_df = subset.sort_values(metric, ascending=ascending).head(n)

        username_col = colmap.get("username")
        country_col = colmap.get("country")

        lines = []
        for _, row in top_df.iterrows():
            name = str(row[username_col]) if username_col else "?"
            val = row[metric]
            country = str(row[country_col]) if country_col else "??"
            lines.append(f"{name} ({country}) — {metric} = {val}")

        header = f"Top {n} joueurs selon **{metric}** :\n\n"
        return header + "\n".join(f"- {ln}" for ln in lines)

    # --- 4. Joueurs d'un pays ---
    if "pays" in q or "country" in q:
        country_col = colmap.get("country")
        if not country_col:
            return "Je ne trouve pas de colonne 'country' dans le dataset."

        # On cherche un code pays en majuscules (FR, JP, US...)
        country_match = re.search(r"\b([A-Z]{2})\b", question)
        if country_match:
            code = country_match.group(1)
            sub = df[df[country_col] == code]
            if sub.empty:
                return f"Aucun joueur trouvé pour le pays **{code}**."
            return (
                f"Il y a **{len(sub):,}** joueurs pour le pays **{code}**.\n"
                "Tu peux aussi demander par exemple : `Top 10 joueurs du FR par pp`."
            ).replace(",", " ")
        else:
            return (
                "Tu peux préciser un code pays en majuscules, par ex. :\n"
                "- `Combien de joueurs du FR ?`\n"
                "- `Top 5 joueurs du JP par pp`"
            )

    # --- 5. Infos sur un joueur ---
    if "joueur" in q or "player" in q or "user" in q or "username" in q:
        username_col = colmap.get("username")
        if not username_col:
            return (
                "Je ne trouve pas de colonne de nom de joueur (`username`).\n"
                "Colonnes dispo : " + ", ".join(df.columns)
            )

        name = None
        quoted = re.search(r'"([^"]+)"', question)
        if quoted:
            name = quoted.group(1)
        else:
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
        for _, row in sub.iterrows():
            base = f"**{row[username_col]}**"
            country_col = colmap.get("country")
            if country_col and country_col in row:
                base += f" ({row[country_col]})"

            extras = []
            for key in ["pp", "rank"]:
                col = colmap.get(key)
                if col and col in row and pd.notna(row[col]):
                    extras.append(f"{col} = {row[col]}")
            if extras:
                base += " — " + ", ".join(extras)

            lines.append(base)

        return "Voici ce que j’ai trouvé :\n\n" + "\n".join(f"- {ln}" for ln in lines)

    # --- 6. Fallback : quelques stats globales ---
    num_cols = df.select_dtypes("number").columns
    text = [
        "Je ne suis pas sûr de comprendre ta question, mais voici quelques infos :",
        f"- Nombre de joueurs : **{len(df):,}**".replace(",", " "),
    ]
    if len(num_cols) > 0:
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
