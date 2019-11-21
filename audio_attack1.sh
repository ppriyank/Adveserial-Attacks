module load cuda/10.1.105
module load cudnn/10.0v7.6.2.24


conda activate audio
cd /home/xl2700/p2/audio_adversarial_examples


python 
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# quit()


for file in short-audio/*.wav
do
  echo "$file"
  python3 classify.py --in $file --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
  python3 attack.py --in $file --target "this is a test" --out /home/xl2700/p2/audio_adversarial_examples/output2/$file --iterations 1000 --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
  python3 classify.py --in $file --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
done
# 
# srun -c4 -t1:00:00 --mem=100000  --pty /bin/bash