#!/bin/bash

ulimit -t unlimited

while getopts o:c: name; do
	case $name in
	  c) config_file=$OPTARG;; 
	  o) res_dir=$OPTARG;;
	esac
done
shift `expr $OPTIND - 1`

if [ "$1" != "" ]; then
	echo "Invalid argument: $1"
	exit
fi

if [ "$res_dir" == "" ]; then
	now_stamp=`date +%Y-%m-%d-%H-%M-%S`
	res_dir=openml_res_$now_stamp
	mkdir $res_dir
fi

if [ "$config_file" == "" ]; then
	config_file="openml-config.json"
fi

cp "$config_file" "$res_dir/$config_file"

python ../genens/genens/tests/run_openml.py --out "$res_dir" --config "$config_file" 2> "$res_dir/err_log" &
py_pid=$!

trap "kill $py_pid 2> /dev/null" EXIT

stopped=""
while kill -0 $py_pid 2> /dev/null; do
	if who | cut -f 1 -d ' ' | grep -qv -f user-exceptions; then
		if [ "$stopped" == "" ]; then
			kill -SIGSTOP $py_pid
			stopped=y
			echo "Process stopped."
		fi
	else
		if [ "$stopped" != "" ]; then
			kill -CONT $py_pid
			stopped=""
			echo "Resumed."
		fi
	fi
	kinit -R
	sleep 30
done

trap - EXIT
