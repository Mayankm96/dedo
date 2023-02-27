# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# get source directory
export SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_PATH

mkdir -p logs

for NENV in 1 5 10 15 20 25 30 35 40 45 50 60 70 80 90 100 110 120 130 140 150 200 250 300 350 400 450 500 600 700 800 900 1000
do
    python test_gym.py --env=HangGarmentRobot-v1 --num_envs=${NENV} --cam_resolution=-1 > logs/output_n_envs_${NENV}.txt
done
