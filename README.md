# Word2Minecraft

![image](https://github.com/JMZ-kk/Word2Minecraft/blob/word2mc_v0/pictures/main_framework_3.png)

This repository contains to code for [Word2Minecraft: Generating 3D Game Levels through Large Language Models].

### Abstract:

We present Word2Minecraft, a system that leverages large language models to generate playable game levels in Minecraft based on structured stories. The system transforms narrative elements—such as protagonist goals, antagonist challenges, and environmental settings—into game levels with both spatial and gameplay constraints. We introduce a flexible framework that allows for the customization of story complexity, enabling dynamic level generation. The system employs a scaling algorithm to maintain spatial consistency while adapting key game elements. We evaluate Word2Minecraft using both metric-based and human-based methods. Our results show that GPT-4-Turbo outperforms GPT-4o-Mini in most areas, including story coherence and objective enjoyment, while the latter excels in aesthetic appeal. We also demonstrate the system’s ability to generate levels with high map enjoyment, offering a promising step forward in the intersection of story generation and game design.

### Usage:

Clone the repo:

`https://github.com/JMZ-kk/Word2Minecraft.git`

Install the environment and activate it:

```
cd Word2World
type > word2world/.env
conda env create -f environment.yml
conda activate word2world
```

Add your API key to the .env file created in word2world folder:

```
OPENAI_API_KEY="sk..."
```

Run with default configs:

`python main.py`

Or run with specified configs:

```
python main.py \
--model="gpt-4-turbo-2024-04-09" \
--min_story_paragraphs=4 \
--max_story_paragraphs=5 \
--total_objectives=8 \
--rounds=1 \
--experiment_name="Your_World" \
--save_dir="outputs"
```

To play the generated game:

```
python word2world/gen_mc_main.py
```
If you meet problems when executing this command and you are using IDEs like Pycharm, try to set the working directory as .../Word2World/word2world


### Results:
#### Buildings:
![image](https://github.com/JMZ-kk/Word2Minecraft/blob/word2mc_v0/pictures/buildings.png)

#### Main maps:
![image](https://github.com/JMZ-kk/Word2Minecraft/blob/word2mc_v0/pictures/037fcbc262e8825faec1132fa71f12b.png)
![image](https://github.com/JMZ-kk/Word2Minecraft/blob/word2mc_v0/pictures/23ea935ed8d849380f7e9463ec62e36.png)
![image](https://github.com/JMZ-kk/Word2Minecraft/blob/word2mc_v0/pictures/dla.png)

### Sub maps:
![image](https://github.com/JMZ-kk/Word2Minecraft/blob/word2mc_v0/pictures/gate.png)
![image](https://github.com/JMZ-kk/Word2Minecraft/blob/word2mc_v0/pictures/gem2.png)
![image](https://github.com/JMZ-kk/Word2Minecraft/blob/word2mc_v0/pictures/mirror%20lake.png)

