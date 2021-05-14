from pathlib import Path

import streamlit as st

from src.pl_modules.ensemble import RelationEnsemble
from src.pl_modules.ger_model import GERModel
from src.pl_modules.rbert_model import RBERT
from src.ui.ui_utils import select_checkpoint


@st.cache(allow_output_mutation=True)
def get_model(checkpoint_path: Path):
    return RelationEnsemble.load_from_checkpoint(checkpoint_path=str(checkpoint_path))


if __name__ == "__main__":
    checkpoint_path = select_checkpoint()
    model: RelationEnsemble = get_model(checkpoint_path=checkpoint_path)
