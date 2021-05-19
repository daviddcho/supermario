# Super Mario Bros

An implementation of a Double Deep Q Network to learn to play Super Mario Bros.

![mario](/mario.png)

## Set Up 
```
# Create virtual environment 
python3 -m venv env

# Activate virtual env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# To deactivate virtual env
deactivate
```

## Usage
```
# Train the agent
./mario train

# Watch mario play 
./mario play
```

## Resources
* Playing Atari with Deep Reinforcement Learning: https://arxiv.org/abs/1312.5602
* Deep Reinforcement Learning with Double Q-learning: https://arxiv.org/abs/1509.06461
* Intro to Reinforcement Learning with David Silver: https://youtube.com/playlist?list=PLqYmG7hTraZBiG_XpjnPrSNw-1XQaM_gB
* Neural Networks from 3Blue1Brown: https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi 
* Conv Nets: A Modular Perspective: https://colah.github.io/posts/2014-07-Conv-Nets-Modular/
