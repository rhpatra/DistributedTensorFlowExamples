#!/bin/bash

# same ip different ports
export ps_host1=localhost
export ps_host2=localhost
export worker1=localhost
export worker2=localhost
export port1=2226
export port2=2227
export port3=2228
export port4=2229

# parameter server 1
python3 example.py --ps_hosts=$ps_host1:$port1,$ps_host2:$port2 --worker_hosts=$worker1:$port3,$worker2:$port4 --job_name=ps 	 --task_index=0 &
# parameter server 2
python3 example.py --ps_hosts=$ps_host1:$port1,$ps_host2:$port2 --worker_hosts=$worker1:$port3,$worker2:$port4 --job_name=ps 	 --task_index=1 &
# worker 1
python3 example.py --ps_hosts=$ps_host1:$port1,$ps_host2:$port2 --worker_hosts=$worker1:$port3,$worker2:$port4 --job_name=worker --task_index=0 &
# worker 2
python3 example.py --ps_hosts=$ps_host1:$port1,$ps_host2:$port2 --worker_hosts=$worker1:$port3,$worker2:$port4 --job_name=worker --task_index=1
