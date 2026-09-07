# Audit du bilinguisme

Date: 2026-09-07

## Conclusion

**Bilinguisme partiel au sens strict du cahier des charges.**

Le rendu français/anglais fonctionne pour les pages et composants testés, mais le projet ne respecte pas encore totalement l'architecture demandee: de nombreux textes restent litteralement codés dans `app.py` et sont traduits par un monkey-patching de Streamlit plutot que par un appel explicite a une fonction `t(key)`.

## Fichiers inspectés

Fichiers Python inspectés: `app.py`, `translations.py`, `train.py`, `pipelines.py`, `prepa.py`, `options/options.py`, `options/option1/indicator.py` et les composants de `options/option1/requirements/`.

Le projet contient aussi `explain.md` et `prompt_bilingue_projet.md`. Aucun fichier HTML, CSS, JavaScript ou page Streamlit supplémentaire n'a été trouvé à la racine active.

## Fichiers modifiés

- `app.py`
- `translations.py`
- `train.py`
- `pipelines.py`
- `prepa.py`
- `bilingual_audit_report.md`
- `options/options.py`
- `options/option1/indicator.py`
- `options/option1/requirements/comprehension_prepa.py`
- `options/option1/requirements/data_quality.py`
- `options/option1/requirements/data_splitting.py`
- `options/option1/requirements/dataset_loading.py`
- `options/option1/requirements/dataset_struct.py`
- `options/option1/requirements/title.py`

## Architecture

- Module central: `translations.py`.
- Langue par défaut: français (`fr`).
- État: `st.session_state["language"]`.
- Choix: `st.sidebar.selectbox` avec `Français` et `English`.
- Rerun: `st.rerun()` lorsqu'un changement de langue est détecté.
- Navigation: clés internes stables (`data_preparation`, `model_training`, etc.).
- Secours: les textes sans traduction restent inchangés.

Limite: `t` reçoit actuellement une paire `(français, anglais)` et non une clé technique unique. Les appels d'interface de `app.py` restent donc codés en dur, même si le wrapper traduit leur rendu.

## Mesures statiques

- Fichiers Python analysés: 13.
- Entrées du registre: 185.
- Appels littéraux d'interface analysés: 95.
- Littéraux sans entrée exacte dans le registre après correction: 0.
- Littéraux identiques dans les deux langues: 4 (`Student Performance Predictor`, `MAE`, `RMSE`, `R²`).
- Couverture des littéraux d'interface par le registre/runtime: 95/95, soit 100%.

Cette mesure ne prouve pas à elle seule que tous les contenus dynamiques sont traduits.

## Couverture par groupe

| Groupe | Français | Anglais | Vérification |
|---|---:|---:|---|
| 1. Préparation, étapes 1-5 | 100% observé | 100% observé | Test navigateur réel |
| 2. Entraînement, étapes 6-9 | 100% des textes statiques mappés | 100% des textes statiques mappés | Vue de l'étape 6 testée |
| 3. Évaluation, étapes 10-13 | 100% des textes statiques mappés | 100% des textes statiques mappés | Vue de l'étape 10 testée |
| 4. Résultats, étapes 14-17 | 100% des textes statiques mappés | 100% des textes statiques mappés | Vue de l'étape 14 testée |
| 5. Rapport, étapes 18-20 | 100% des textes statiques mappés | 100% des textes statiques mappés | Vue de l'étape 18 testée |

Les pourcentages ci-dessus concernent les textes recensés et mappés, pas une preuve exhaustive de chaque état conditionnel.

## Tableaux, métadonnées et graphiques

- Les colonnes des `DataFrame` sont renommées dans une copie destinée à l'affichage anglais.
- Les noms de modèles (`dummy_mean`, `ridge`, etc.) sont présentés en anglais dans les copies de tableaux.
- Les clés de métadonnées JSON sont traduites dans l'affichage.
- Les titres et axes des figures de `train.py`, `pipelines.py` et `prepa.py` utilisent `t(...)`.
- Les noms internes du modèle et les colonnes originales restent inchangés pour les calculs.

## Tests réalisés

### Français

- Français sélectionné par défaut: réussi.
- Navigation du groupe 1: réussie.
- Affichage des indicateurs, tableaux et messages du groupe 1: réussi.
- Affichage initial des groupes 2, 3, 4 et 5: réussi dans le test navigateur.

### Anglais

- Passage à English: réussi.
- Traduction de la sidebar, du titre, de l'introduction, des indicateurs et du groupe 1: réussie.
- Navigation affichée en anglais pour les groupes 2 à 5: réussie.
- Vues initiales des étapes 6, 10, 14 et 18: observées en anglais.
- Traduction des colonnes de tableaux et métadonnées: test Python direct réussi.

### Conservation de l'état

- Le routage utilise des clés internes stables et ne dépend plus du texte traduit.
- Le changement de langue appelle `st.rerun()` sans supprimer `st.session_state`.
- Le scénario complet comparaison + prédiction + graphique + changement de langue n'a pas été entièrement automatisé dans cette session.

## Problèmes restant à résoudre

1. `app.py` contient encore des textes français littéraux dans les appels `st.title`, `st.write`, `st.button`, etc. Le wrapper les traduit à l'exécution, mais ils ne passent pas explicitement par `t(key)`.
2. Le système de clés explicites demandé (`t("group.training.title")`) n'est pas encore en place.
3. Aucun outil automatisé permanent ne signale les nouvelles clés manquantes dans le CI.
4. Le serveur Streamlit de test a refusé une reconnexion après redémarrage; la validation finale de tous les clics conditionnels n'est donc pas déclarée complète.
5. Les contrôles Streamlit natifs comme `Show/hide columns`, `Download as CSV`, `Search` et `Fullscreen` restent dans la langue de Streamlit/browser, pas dans le registre du projet.

## Validation technique

- `py_compile` de tous les fichiers Python actifs: réussi.
- Import complet de `app.py`: réussi (`platform ready`).
- Test direct du registre, des colonnes, des noms de modèles et des métadonnées: réussi.
- Référence `SPP` inexistante dans le graphique historique de `pipelines.py`: corrigée au profit de `importances`.
- Commande de lancement:

```bash
.venv/bin/streamlit run app.py
```
