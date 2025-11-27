# -*- coding: utf-8 -*-
"""
Chatbot osu! standard players (2007 -> today)

- Lit les données depuis :
  /kaggle/input/osustandard-players-from-2007-to-today/2025_05_16_players_info.csv
- Interface de chat avec Streamlit
- Répond à des questions simples sur les joueurs / pays / top rankings
"""

import os
import re

import pandas as pd
import streamlit as st


# ================== CONFIG ==================

# Chemin Kaggle par défaut (tu peux le changer si besoin)
DATA_PATH = "/kaggle/input/osustandard-players-from-2007-to-today/2025_05_16_players_info.csv"


# ================== CHARGEMENT DES DONNÉES ==================

@st.cache_data(show_spinner=True)
def load_players(path: str) -> pd.DataFrame:
    """Charge le CSV des joueurs."""
    if not os.path.exists(path):
        st.error(f"Fichier introuvable : {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Normalise quelques noms de colonnes possibles (à adapter si nécessaire)
    cols = {c.lower(): c for c in df.columns}  # map lowercase -> original

    # On essaie d'identifier ces colonnes si elles existent
    username_col = cols.get("username") or cols.get("user") or cols.get("name")
    country_col = cols.get("country") or cols.get("country_code")
    pp_col = cols.get("pp") or cols.get("performance_points")
    rank_col = cols.get("global_rank") or cols.get("rank")

    return df, {
        "username": username_col,
        "country": country_col,
        "pp": pp_col,
        "rank": rank_col,
    }


# ================== LOGIQUE DU CHATBOT ==================

def answer_question(question: str, df: pd.DataFrame, colmap: dict) -> str:
    """
    Génère une réponse simple à partir de la question et du DataFrame.
    Pas d'IA ici : juste des règles + pandas.
    """
    if df.empty:
        return "Je n'ai pas pu charger les données des joueurs. Vérifie le chemin du CSV."

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

        # Choix de la métrique
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

        # Pour un rang, on veut le plus petit ; pour les pp, le plus grand
        ascending = "rank" in metric.lower()
        top_df = subset.sort_values(metric, ascending=ascending).head(n)

        username_col = colmap.get("username")
        country_col = colmap.get("country")

        lines = []
        for i, row in top_df.iterrows():
            name = str(row[username_col]) if username_col else f"id={i}"
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

        # On suppose un code pays en majuscules (ex : JP, FR, US)
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

        # On essaie de récupérer un nom :
        #   - texte entre guillemets "..."
        #   - ou tout après le mot "joueur"/"player"
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

        # On limite aux 3 premiers résultats
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

    # --- 6. Question générique : on renvoie quelques stats de base ---
    num_cols = df.select_dtypes("number").columns
    text = [
        f"Je ne suis pas sûr de comprendre ta question, mais voici quelques infos globales :",
        f"- Nombre de joueurs : **{len(df):,}**".replace(",", " "),
    ]
    if len(num_cols) > 0:
        # On prend 3 colonnes numériques pour résumer
        for col in num_cols[:3]:
            text.append(
                f"- {col} : min={df[col].min():.2f}, max={df[col].max():.2f}, moyenne={df[col].mean():.2f}"
            )

    text.append(
        "\nTu peux demander : `help` ou `aide` pour voir des exemples de questions."
    )
    return "\n".join(text)


# ================== APP STREAMLIT ==================

def main():
    st.set_page_config(page_title="Chatbot osu! players", page_icon="🎧")

    st.title("🎧 Chatbot osu! standard players (2007 → aujourd’hui)")
    st.write(
        "Je réponds à des questions sur les joueurs à partir du fichier CSV Kaggle.\n"
        "Pose ta question en bas. Tu peux taper `help` pour voir des exemples."
    )

    # Chargement des données
    with st.spinner("Chargement des données des joueurs..."):
        df, colmap = load_players(DATA_PATH)

    if df.empty:
        st.stop()

    # Petit résumé dans la sidebar
    st.sidebar.header("Données")
    st.sidebar.write(f"**Lignes :** {len(df):,}".replace(",", " "))
    st.sidebar.write(f"**Colonnes :** {len(df.columns)}")
    st.sidebar.write("Colonnes principales détectées :")
    for k, v in colmap.items():
        st.sidebar.write(f"- {k} → {v}")

    st.sidebar.caption("Si un nom de colonne ne correspond pas, adapte le code dans `load_players`.")

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
                    "- `Combien de joueurs du JP ?`\n"
                ),
            }
        ]

    # Affichage de l'historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Entrée utilisateur
    if prompt := st.chat_input("Pose une question sur les joueurs osu!..."):
        # Ajout du message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Génération de la réponse
        answer = answer_question(prompt, df, colmap)

        # Ajout de la réponse
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Réafficher tout
        st.experimental_rerun()


if __name__ == "__main__":
    main()
