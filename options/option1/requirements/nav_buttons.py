import streamlit as st

def go_to_step(step_index: int) -> None:
    """Affiche l'étape demandée après le clic sur un bouton."""
    st.session_state.data_step = step_index
    st.rerun()

# Fonction pour changer d'étape
def nav_buttons(step_names: list[str], go_to_step: callable) -> None:
    
    nav_columns = st.columns(len(step_names))

    for index, step_name in enumerate(step_names):
        with nav_columns[index]:
            if st.session_state.data_step == index:
                st.button(
                    f"Étape {index + 1}",
                    key=f"current_step_{index}",
                    disabled=True,
                    width="stretch",
                )
            else:
                st.button(
                    f"Étape {index + 1}",
                    key=f"step_button_{index}",
                    on_click=go_to_step,
                    args=(index,),
                    width="stretch",
                )