#!/bin/bash

n_times=1
while getopts o:c:n: name; do
	case $name in
	  c) config_file=$OPTARG;; 
	  o) res_dir=$OPTARG;;
	  n) n_times=$OPTARG;;
	esac
done
shift `expr $OPTIND - 1`

if [ "$1" != "" ]; then
	echo "Invalid argument: $1"
	exit
fi

if [ "$config_file" == "" ]; then
	echo "Missing config file"
	exit 1
fi

if [ "$res_dir" == "" ]; then
	now_stamp=`date +%Y-%m-%d-%H-%M-%S`
	res_dir=res_$now_stamp
	mkdir $res_dir
fi

cp "$config_file" "$res_dir/config.json"

for i in $(eval echo "{1..$n_times}"); do
	if [ -d "$res_dir/$i" ] ; then
		echo "Skipping test $i"
		continue
	else
		mkdir "$res_dir/$i"
		out_dir="$res_dir/$i"
	fi

	python ../genens/genens/tests/run_datasets.py --out "$out_dir" config "$config_file" 2> "$out_dir/err_log" &
	py_pid=$!

	trap "kill $py_pid 2> /dev/null" EXIT

	while kill -0 $py_pid 2> /dev/null; do
		sleep 30
	done

	trap - EXIT
done
