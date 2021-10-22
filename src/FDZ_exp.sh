# Evaluation for the "Fusion Dead Zone" experiment. 
# e.g. original, blackout_R, blackout_T, SidesBlackout_a, SidesBlackout_b, SurroundingBlackout

python inference.py --FDZ original --model-path ../pretrained/best_checkpoint.pth.tar &&
python inference.py --FDZ blackout_R --model-path ../pretrained/best_checkpoint.pth.tar &&
python inference.py --FDZ blackout_T --model-path ../pretrained/best_checkpoint.pth.tar &&
python inference.py --FDZ SidesBlackout_a --model-path ../pretrained/best_checkpoint.pth.tar &&
python inference.py --FDZ SidesBlackout_b --model-path ../pretrained/best_checkpoint.pth.tar &&
python inference.py --FDZ SurroundingBlackout --model-path ../pretrained/best_checkpoint.pth.tar