#!/bin/bash

ulimit -t unlimited

while getopts o:c: name; do
	case $name in
	  c) config_file=$OPTARG;; 
	  o) res_dir=$OPTARG;;
	esac
done
shift `expr $OPTIND - 1`


if [ "$res_dir" == "" ]; then
	now_stamp=`date +%Y-%m-%d-%H-%M-%S`
	res_dir=res_$now_stamp
	mkdir $res_dir
fi

if [ "$config_file" == "" ]; then
	config_file="config.json"
fi

python ../genens/genens/tests/run_datasets.py --out $res_dir config $config_file &
py_pid=$!

trap "kill $py_pid 2> /dev/null" EXIT

stopped=""
while kill -0 $py_pid 2> /dev/null; do
	if who | cut -f 1 -d ' ' | grep -qv "suchopag"; then
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
	sleep 30
done

trap - EXIT
