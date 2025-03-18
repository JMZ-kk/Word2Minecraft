# Word2Minecraft

![image](https://github.com/JMZ-kk/Word2Minecraft/blob/main/picture/main_framework_3.png)

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
python word2world/play_game.py "path_to_game_data\game_data.json"
```
where `game_data.json` is generated when the Word2World loop is finished and is saved to `\outputs\game_data.json`. This can be modified in `configs` or as `--save_dir` arg.

To play an example world:

```
python word2world/play_game.py
```

### Results:

#### LLM comparison:

![image](https://github.com/umair-nasir14/Word2World/assets/68095790/7b843e04-d009-4708-9b3e-686ddfe9c358)

#### Worlds:

![Untitled design](https://github.com/umair-nasir14/Word2World/assets/68095790/de351d5b-a8bf-4f11-8af9-a8eee1e45c33)

![world_1](https://github.com/umair-nasir14/Word2World/assets/68095790/5b85bb03-eed4-4879-ab07-4683d317ab20)
![world_2](https://github.com/umair-nasir14/Word2World/assets/68095790/6ccaa7e3-6573-4f20-b3a9-03e8992ffc9c)
![world_3](https://github.com/umair-nasir14/Word2World/assets/68095790/53e38643-d10a-4c16-a584-c0aa19116e60)
![world_4](https://github.com/umair-nasir14/Word2World/assets/68095790/fc8df4a5-63db-414f-96ca-a4094397ff9d)
![world_6](https://github.com/umair-nasir14/Word2World/assets/68095790/d92fa869-82de-4e97-bb77-2eb5fb7d04e2)
![world_7](https://github.com/umair-nasir14/Word2World/assets/68095790/751a753e-9e3d-41da-b146-fa852d0e7f1c)

### Note:

- The most stable model is `"gpt-4-turbo-2024-04-09"`.
- Currently only `OpenAI` models are supported.
- OS supported: `Windows`

### To-dos:

- [ ] Add support for Anthropic.
- [ ] Add support for Groq.
- [ ] Add support for Linux.
- [ ] Clean Code for easy integrations of new platforms, e.g. huggingface.

### Cite:
```
@article{nasir2024word2world,
  title={Word2World: Generating Stories and Worlds through Large Language Models},
  author={Nasir, Muhammad U and James, Steven and Togelius, Julian},
  journal={arXiv preprint arXiv:2405.06686},
  year={2024}
}
```

