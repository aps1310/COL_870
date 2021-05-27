if [ $# == 4 ]
then
  python3 run_gan.py --query_path "$1" --path_to_sample_images "$2" --gen_images "$3" --gen_labels  "$4"
else
  python3 run_gan.py --query_path "$1" --path_to_sample_images "$2" --gen_images "$3" --gen_labels  "$4" --model_path "$5"
fi