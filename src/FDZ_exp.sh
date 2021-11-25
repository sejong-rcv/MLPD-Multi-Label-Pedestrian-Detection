# Evaluation for the "Fusion Dead Zone" experiment. 
# e.g. original, blackout_R, blackout_T, SidesBlackout_a, SidesBlackout_b, SurroundingBlackout

python inference.py --FDZ original --model-path ../pretrained/best_checkpoint.pth.tar &&
python inference.py --FDZ blackout_r --model-path ../pretrained/best_checkpoint.pth.tar &&
python inference.py --FDZ blackout_t --model-path ../pretrained/best_checkpoint.pth.tar &&
python inference.py --FDZ sidesblackout_a --model-path ../pretrained/best_checkpoint.pth.tar &&
python inference.py --FDZ sidesblackout_b --model-path ../pretrained/best_checkpoint.pth.tar &&
python inference.py --FDZ surroundingblackout --model-path ../pretrained/best_checkpoint.pth.tar