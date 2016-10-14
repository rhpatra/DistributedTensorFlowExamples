#!/bin/bash
# for aws ec2, the following ips are all "private ip". 
ps_host1=172.31.54.141
ps_host2=172.31.53.135
worker1=172.31.63.7
worker2=172.31.53.190
port=2222

ip="$(hostname --ip-address)"

if [ ${ip} == ${ps_host1} ]; then
	# launch the code on each machine 
	python3 distributed_example.py --ps_hosts=$ps_host1:$port,$ps_host2:$port --worker_hosts=$worker1:$port,$worker2:$port --job_name=ps --task_index=0
fi

if [ ${ip} == ${ps_host2} ]; then
	python3 distributed_example.py --ps_hosts=$ps_host1:$port,$ps_host2:$port --worker_hosts=$worker1:$port,$worker2:$port --job_name=ps --task_index=1 
fi

if [ ${ip} == ${worker1} ]; then
	python3 distributed_example.py --ps_hosts=$ps_host1:$port,$ps_host2:$port --worker_hosts=$worker1:$port,$worker2:$port --job_name=worker --task_index=0
fi

if [ ${ip} == ${worker2} ]; then
	python3 distributed_example.py --ps_hosts=$ps_host1:$port,$ps_host2:$port --worker_hosts=$worker1:$port,$worker2:$port --job_name=worker --task_index=1
fi
