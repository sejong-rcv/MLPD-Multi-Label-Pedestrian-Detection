# Evaluation for the "Fusion Dead Zone" experiment. 
# e.g. original, blackout_R, blackout_T, SidesBlackout_a, SidesBlackout_b, SurroundingBlackout

python eval.py &&
python eval.py --FDZ blackout_R &&
python eval.py --FDZ blackout_T &&
python eval.py --FDZ SidesBlackout_a &&
python eval.py --FDZ SidesBlackout_b &&
python eval.py --FDZ SurroundingBlackout
