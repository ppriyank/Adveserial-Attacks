conda activate audio
cd /home/xl2700/p2/audio_adversarial_examples


for file in ~/short-audio/*.wav
do
  echo "$file"
  python3 classify.py --in $file --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
  python3 attack.py --in $file --target "this is a test" --out $file --iterations 1000 --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
  python3 classify.py --in $file --restore_path deepspeech-0.4.1-checkpoint/model.v0.4.1
done

