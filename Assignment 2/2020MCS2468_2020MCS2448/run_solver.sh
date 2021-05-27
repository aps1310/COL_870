if [ $# == 4 ]
then
  python3 query_to_symbolic_solution.py --train_dir "$1" --test_dir "$2" --output_file "$4"
else
  python3 query_to_symbolic_solution.py --train_dir "$1" --test_dir "$2" --output_file "$4"  --model_path "$5"
fi
