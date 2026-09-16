"""Photo-realistic distortion helpers and crop-level mark calibration (no PHI)."""

from med_doc.eval.bakeoff import text_bakeoff, vision_bakeoff
from med_doc.eval.calibrate_marks import fit_mark_logreg, train_and_save
from med_doc.eval.crop_qa import crop_qa, diagnose_crop
from med_doc.eval.export_crops import export_labeled_crops
from med_doc.eval.mark_gold import GoldSet, SheetGold, load_gold
from med_doc.eval.models import models_in_use
from med_doc.eval.photoreal import distort_sheet, draw_ticks, shadow_gradient

__all__ = [
    "text_bakeoff",
    "vision_bakeoff",
    "GoldSet",
    "SheetGold",
    "crop_qa",
    "diagnose_crop",
    "distort_sheet",
    "draw_ticks",
    "export_labeled_crops",
    "fit_mark_logreg",
    "load_gold",
    "models_in_use",
    "shadow_gradient",
    "train_and_save",
]
