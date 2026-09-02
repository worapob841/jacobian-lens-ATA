# rsync rzvh --progress --partial -e "ssh -p 6022" orachat.c@localhost:/g/home/orachat.c/project/MLLM/jacobian-lens-ATA/ ./
# srun --partition=UniNet -w compute01 --qos uninet-limit64cores500gb --cpus-per-task=8 --mem=16G --pty bash 
# rsync -rzvh --progress --partial --append-verify -e "ssh -J diplab@161.246.5.159:6022" ./ diplab@192.168.1.104:/volume1/worapob-research/MLLM/jacobian-lens-ATA
# rsync -rzvh --progress --partial -e "ssh -J diplab@161.246.5.159:6022" diplab@192.168.1.104:/volume1/worapob-research/MLLM/jacobian-lens-ATA/ ./