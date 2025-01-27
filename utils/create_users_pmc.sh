#!/bin/bash

create_user_no_prompt () {
	echo ========$1
	echo ========$2

	local USERNAME="$1"
	local PASSWORD="$2"
	local GROUP="$3"
	local FULLNAME="$4"
	local EMAIL="$5"

	local GECOS=""
	echo "Creating user ${USERNAME} ${FULLNAME} ${EMAIL}..."
	adduser --ingroup ${GROUP} --disabled-login --gecos "" ${USERNAME}
	echo "Setting password..."
	echo "${USERNAME}:${PASSWORD}" | chpasswd
	echo "Adding user to docker group..."
	usermod -aG docker ${USERNAME}
	echo "Running make..."
	( cd /var/yp/; make)
}

[ $SUDO_USER ] && user=$SUDO_USER || user=$(whoami)

GROUP="studenti_psmc"

while IFS=$'\t' read -r EMAIL PASSWORD NAME SURNAME; do
    USERNAME=$(echo $EMAIL | cut -d '@' -f 1 | tr '.' '_')
    FULLNAME="${NAME} ${SURNAME}"
    #echo "----"
    #echo ${USERNAME}
    #echo ${EMAIL}
    #echo ${PASSWORD}
    #echo ${GROUP}
    #echo ${FULLNAME}
    #echo ${EMAIL}
    create_user_no_prompt "${USERNAME}" "${PASSWORD}" "${GROUP}" "${FULLNAME}" "${EMAIL}"
done < users_list.csv
