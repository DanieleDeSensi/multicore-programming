#!/usr/bin/env bash

##**************************************************************
##
## Copyright (C) 1990-2018, Condor Team, Computer Sciences Department,
## University of Wisconsin-Madison, WI.
##
## Licensed under the Apache License, Version 2.0 (the "License"); you
## may not use this file except in compliance with the License.  You may
## obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
##**************************************************************

# This is a script to run OpenMPI jobs under the HTCondor parallel universe.
# OpenMPI assumes that a full install is available on all execute nodes.

## sample submit script
#universe = parallel
#executable = openmpiscript
#arguments = actual_mpi_job arg1 arg2 arg3
#getenv = true
#
#should_transfer_files = yes
#transfer_input_files = actual_mpi_job
#when_to_transfer_output = on_exit_or_evict
#
#output = out.$(NODE)
#error  = err.$(NODE)
#log    = log
#
#machine_count = 8
#queue
##

## configuration options
# $USE_OPENMP should be set to true if using OpenMP with your OpenMPI executable (not typical).
USE_OPENMP=false

# Set the paths to the helper scripts
# Get them from the HTCondor libexec directory
ORTED_LAUNCHER=$(condor_config_val libexec)/orted_launcher.sh
GET_ORTED_CMD=$(condor_config_val libexec)/get_orted_cmd.sh
# Or set a custom path (e.g. the local directory if transferring the scripts)
#ORTED_LAUNCHER=./orted_launcher.sh
#GET_ORTED_CMD=./get_orted_cmd.sh

# $MPDIR points to the location of the OpenMPI install
# The pool admin may set it via OPENMPI_INSTALL_PATH in the condor_config (recommended)


MPDIR=/usr
#MPDIR=$(condor_config_val OPENMPI_INSTALL_PATH)





# Or set it manually
#MPDIR=/usr/lib64/openmpi

# $EXINT is a comma-delimited list of excluded network interfaces.
# If your mpi jobs are hanging, OpenMPI may be trying to use too many
# network interfaces to communicate between nodes.
# The pool admin may set it via OPENMPI_EXCLUDE_NETWORK_INTERFACES in the condor_config (recommended)
EXINT=$(condor_config_val OPENMPI_EXCLUDE_NETWORK_INTERFACES)
# Or set it manually
#EXINT="docker0,virbr0"
##

## configuration check
# We recommend that your pool admin use MOUNT_UNDER_SCRATCH = /tmp
# so that OpenMPI caches all data under the user's scratch directory.
# Not having /tmp mounted under scratch may hang mpi jobs.
_USE_SCRATCH=$(condor_config_val MOUNT_UNDER_SCRATCH)
if [ -z $_USE_SCRATCH ]; then
    >&2 echo "WARNING: MOUNT_UNDER_SCRATCH not set in condor_config"
elif test "${_USE_SCRATCH#*/tmp}" == "$_USE_SCRATCH"; then
    >&2 echo "WARNING: /tmp not included in MOUNT_UNDER_SCRATCH"
fi

# If MPDIR is not set, then use a default value
if [ -z $MPDIR ]; then
    >&2 echo "WARNING: Using default value for \$MPDIR in openmpiscript"
    MPDIR=/usr/lib64/openmpi
fi
PATH=$MPDIR/bin:.:$PATH
export PATH

# If EXINT is not set, then use some default values
if [ -z $EXINT ]; then
    >&2 echo "WARNING: Using default values for \$EXINT in openmpiscript"
    EXINT="docker0,virbr0"
fi
##

## cleanup function
_orted_launcher_pid=0
_mpirun_pid=0
CONDOR_CHIRP=$(condor_config_val libexec)/condor_chirp
force_cleanup() {
    # Forward SIGTERM to the orted launcher
    if [ $_orted_launcher_pid -ne 0 ]; then
	kill -s SIGTERM $_orted_launcher_pid
    fi

    # Cleanup mpirun
    if [ $_CONDOR_PROCNO -eq 0 ] && [ $_mpirun_pid -ne 0 ]; then
	$CONDOR_CHIRP ulog "Node $_CONDOR_PROCNO caught SIGTERM, cleaning up mpirun"
	rm $HOSTFILE
	
	# Send SIGTERM to mpirun and the orted launcher
	kill -s SIGTERM $_mpirun_pid

	# Give mpirun 30 seconds to terminate nicely
	for i in {1..30}; do
	    kill -0 $_mpirun_pid 2> /dev/null # returns 0 if running
	    _mpirun_killed=$?
	    if [ $_mpirun_killed -ne 0 ]; then
		break
	    fi
	    sleep 1
	done

	# If mpirun is still running, send SIGKILL
	if [ $_mpirun_killed -eq 0 ]; then
	    $CONDOR_CHIRP ulog "mpirun hung on Node ${_CONDOR_PROCNO}, sending SIGKILL!"
	    kill -s SIGKILL $_mpirun_pid
	fi

    fi
    exit 1
}
trap force_cleanup SIGTERM
##

## execute node setup
export PATH=$MPDIR/bin:$PATH

# Run the orted launcher (gets orted command from condor_chirp)
$ORTED_LAUNCHER &
_orted_launcher_pid=$!
if [ $_CONDOR_PROCNO -ne 0 ]; then
    # If not on node 0, wait for orted
    wait $_orted_launcher_pid
    exit $?
fi
##

## head node (node 0) setup
# Build the hostfile
HOSTFILE=hosts
while [ -f $_CONDOR_SCRATCH_DIR/$HOSTFILE ]; do
    HOSTFILE=x$HOSTFILE
done
HOSTFILE=$_CONDOR_SCRATCH_DIR/$HOSTFILE
REQUEST_CPUS=$(condor_q -jobads $_CONDOR_JOB_AD -af RequestCpus)

for node in $(seq 0 $(( $_CONDOR_NPROCS - 1 ))); do
    if $USE_OPENMP; then
	# OpenMP will do the threading on the execute node
	echo "$node slots=1" >> $HOSTFILE
    else
	# OpenMPI will do the threading on the execute node
	echo "$node slots=$REQUEST_CPUS" >> $HOSTFILE
    fi
done

# Make sure the executable is executable
EXECUTABLE=$1
shift
chmod +x $EXECUTABLE
##

## run mpirun
# Set MCA values for running on HTCondor
export OMPI_MCA_plm_rsh_agent=$GET_ORTED_CMD     # use the helper script instead of ssh
export OMPI_MCA_plm_rsh_no_tree_spawn=1          # disable ssh tree spawn
export OMPI_MCA_orte_hetero_nodes=1              # do not assume same hardware on each node
export OMPI_MCA_orte_startup_timeout=120         # allow two minutes before failing
export OMPI_MCA_hwloc_base_binding_policy="none" # do not bind to cpu cores
export OMPI_MCA_btl_tcp_if_exclude="lo,$EXINT"   # exclude unused tcp network interfaces

# Optional MCA values to set for firewalled setups
#export OMPI_MCA_btl_tcp_port_min_v4=1024    # lowest port number that can be used
#export OMPI_MCA_btl_tcp_port_range_v4=64511 # range of ports above lowest that can be used

# Optionally set MCA values for increasing mpirun verbosity per component
# (see ompi_info for more components)
#export OMPI_MCA_plm_base_verbose=30
#export OMPI_MCA_orte_base_verbose=30
#export OMPI_MCA_hwloc_base_verbose=30
#export OMPI_MCA_btl_base_verbose=30

# Run mpirun in the background and wait for it to exit
mpirun -v --prefix $MPDIR -hostfile $HOSTFILE $EXECUTABLE $@ &
_mpirun_pid=$!
wait $_mpirun_pid
_mpirun_exit=$?

## clean up
# Wait for orted to finish
wait $_orted_launcher_pid
rm $HOSTFILE
exit $_mpirun_exit
