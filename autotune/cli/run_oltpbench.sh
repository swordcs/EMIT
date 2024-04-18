#!/usr/bin/env bash
cd /home/gengj/Project/${1} && java -jar benchbase.jar -b ${2} -c ${3} -s 1 -d ${4} ${5} ${6}
