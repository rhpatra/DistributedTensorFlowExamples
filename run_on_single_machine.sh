#!/bin/bash

# same ip different ports
ps_host1=localhost
ps_host2=localhost
worker1=localhost
worker2=localhost
port1=2226
port2=2227
port3=2228
port4=2229

# parameter server 1
python3 distributed_example.py --ps_hosts=$ps_host1:$port1,$ps_host2:$port2 --worker_hosts=$worker1:$port3,$worker2:$port4 --job_name=ps 	 --task_index=0 &
# parameter server 2
python3 distributed_example.py --ps_hosts=$ps_host1:$port1,$ps_host2:$port2 --worker_hosts=$worker1:$port3,$worker2:$port4 --job_name=ps 	 --task_index=1 &
# worker 1
python3 distributed_example.py --ps_hosts=$ps_host1:$port1,$ps_host2:$port2 --worker_hosts=$worker1:$port3,$worker2:$port4 --job_name=worker --task_index=0 &
# worker 2
python3 distributed_example.py --ps_hosts=$ps_host1:$port1,$ps_host2:$port2 --worker_hosts=$worker1:$port3,$worker2:$port4 --job_name=worker --task_index=1
